"""GPU backend for the ModelFit prototype, deployed to Modal.

`run_trial` is plain Python so it can be tested locally on CPU with a small
slice of data. `train_and_eval` is the Modal function that runs it on a GPU.
"""

import json
import time
import uuid

import modal

app = modal.App("modelfit")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "scikit-learn",
)

# Trained models and per-trial results are kept here for later phases.
artifacts = modal.Volume.from_name("modelfit-artifacts", create_if_missing=True)
ARTIFACTS_DIR = "/artifacts"

DEFAULT_HPARAMS = {
    "epochs": 3,
    "learning_rate": 5e-5,
    "batch_size": 32,
    "max_length": 64,
}


def run_trial(
    model_id: str,
    dataset_id: str = "PolyAI/banking77",
    hparams: dict | None = None,
    max_train_rows: int | None = None,
    max_eval_rows: int | None = None,
    output_dir: str = "/tmp/modelfit-trial",
) -> dict:
    """Fine-tune one model on one dataset and report quality metrics."""
    import numpy as np
    import torch
    from datasets import load_dataset
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    hp = {**DEFAULT_HPARAMS, **(hparams or {})}

    ds = load_dataset(dataset_id)
    train, test = ds["train"], ds["test"]
    if max_train_rows:
        train = train.shuffle(seed=0).select(range(min(max_train_rows, len(train))))
    if max_eval_rows:
        test = test.shuffle(seed=0).select(range(min(max_eval_rows, len(test))))
    label_names = ds["train"].features["label"].names
    num_labels = len(label_names)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=hp["max_length"])

    train = train.map(tokenize, batched=True, remove_columns=["text"])
    test = test.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )

    use_gpu = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hp["epochs"],
        learning_rate=hp["learning_rate"],
        per_device_train_batch_size=hp["batch_size"],
        per_device_eval_batch_size=128,
        fp16=use_gpu,
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    start = time.time()
    trainer.train()
    train_seconds = time.time() - start

    pred = trainer.predict(test)
    y_pred = np.argmax(pred.predictions, axis=-1)
    y_true = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    trainer.save_model(f"{output_dir}/model")
    tokenizer.save_pretrained(f"{output_dir}/model")

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "hparams": hp,
        "num_labels": num_labels,
        "train_rows": len(train),
        "eval_rows": len(test),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "train_seconds": round(train_seconds, 1),
        "device": torch.cuda.get_device_name(0) if use_gpu else "cpu",
    }


@app.function(gpu="L4", image=image, volumes={ARTIFACTS_DIR: artifacts}, timeout=30 * 60)
def train_and_eval(model_id: str, dataset_id: str = "PolyAI/banking77", hparams: dict | None = None) -> dict:
    trial_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    trial_dir = f"{ARTIFACTS_DIR}/trials/{trial_id}"
    result = run_trial(model_id, dataset_id, hparams, output_dir=trial_dir)
    result["trial_id"] = trial_id
    with open(f"{trial_dir}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    artifacts.commit()
    return result


@app.function(gpu="L4", image=modal.Image.debian_slim().pip_install("torch"))
def gpu_check():
    import torch
    return f"GPU: {torch.cuda.get_device_name(0)}, CUDA available: {torch.cuda.is_available()}"
