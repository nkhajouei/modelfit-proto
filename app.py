import time

import gradio as gr
import modal
import spaces

# Models offered in the trial dropdown. Phase 1 is tested with distilbert.
MODELS = [
    "distilbert-base-uncased",
]
DATASETS = [
    "PolyAI/banking77",
]


# ZeroGPU Spaces refuse to start unless at least one @spaces.GPU function exists.
# Our real GPU work runs on Modal; this placeholder only satisfies that check
# and is never called.
@spaces.GPU
def _zerogpu_placeholder():
    return None


def check_gpu():
    fn = modal.Function.from_name("modelfit", "gpu_check")
    return fn.remote()


def run_trial(model_id, dataset_id, epochs, learning_rate):
    """Start a training trial on Modal and stream progress back to the page."""
    fn = modal.Function.from_name("modelfit", "train_and_eval")
    hparams = {"epochs": int(epochs), "learning_rate": float(learning_rate)}
    call = fn.spawn(model_id, dataset_id, hparams)

    start = time.time()
    while True:
        try:
            result = call.get(timeout=5)
            break
        except TimeoutError:
            elapsed = int(time.time() - start)
            msg = (
                f"Training **{model_id}** on **{dataset_id}** on a cloud GPU... "
                f"{elapsed // 60}m {elapsed % 60:02d}s elapsed. "
                "The first run takes a few extra minutes while the GPU image is built."
            )
            yield msg, None

    total = int(time.time() - start)
    rows = [
        ["Model", result["model_id"]],
        ["Dataset", result["dataset_id"]],
        ["Accuracy", f"{result['accuracy']:.3f}"],
        ["Precision (macro)", f"{result['precision_macro']:.3f}"],
        ["Recall (macro)", f"{result['recall_macro']:.3f}"],
        ["F1 (macro)", f"{result['f1_macro']:.3f}"],
        ["Training time", f"{result['train_seconds']:.0f} s"],
        ["Train / eval rows", f"{result['train_rows']:,} / {result['eval_rows']:,}"],
        ["Classes", str(result["num_labels"])],
        ["GPU", result["device"]],
        ["Trial ID", result["trial_id"]],
    ]
    yield f"Done in {total // 60}m {total % 60:02d}s.", rows


with gr.Blocks(title="ModelFit") as demo:
    gr.Markdown("# ModelFit prototype\nBring your constraints, we find the tradeoff.")

    with gr.Tab("Run a trial"):
        gr.Markdown(
            "Fine-tune one model on one dataset on a cloud GPU and see how good it is."
        )
        with gr.Row():
            model_in = gr.Dropdown(MODELS, value=MODELS[0], label="Model")
            dataset_in = gr.Dropdown(DATASETS, value=DATASETS[0], label="Dataset")
        with gr.Row():
            epochs_in = gr.Slider(1, 5, value=3, step=1, label="Epochs")
            lr_in = gr.Dropdown(
                [2e-5, 3e-5, 5e-5, 1e-4], value=5e-5, label="Learning rate"
            )
        run_btn = gr.Button("Run trial", variant="primary")
        status = gr.Markdown()
        results = gr.Dataframe(headers=["Metric", "Value"], label="Results", interactive=False)
        run_btn.click(
            run_trial,
            inputs=[model_in, dataset_in, epochs_in, lr_in],
            outputs=[status, results],
        )

    with gr.Tab("System check"):
        btn = gr.Button("Check cloud GPU")
        out = gr.Textbox(label="Result")
        btn.click(check_gpu, outputs=out)

demo.launch()
