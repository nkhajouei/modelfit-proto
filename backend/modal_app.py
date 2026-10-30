import modal

app = modal.App("modelfit")
image = modal.Image.debian_slim().pip_install("torch")


@app.function(gpu="L4", image=image)
def gpu_check():
    import torch
    return f"GPU: {torch.cuda.get_device_name(0)}, CUDA available: {torch.cuda.is_available()}"
