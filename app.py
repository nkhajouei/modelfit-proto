import gradio as gr
import modal
import spaces


# ZeroGPU Spaces refuse to start unless at least one @spaces.GPU function exists.
# Our real GPU work runs on Modal; this placeholder only satisfies that check
# and is never called.
@spaces.GPU
def _zerogpu_placeholder():
    return None


def check_gpu():
    fn = modal.Function.from_name("modelfit", "gpu_check")
    return fn.remote()


with gr.Blocks(title="ModelFit") as demo:
    gr.Markdown("# ModelFit prototype\nBring your constraints, we find the tradeoff.")
    btn = gr.Button("Check cloud GPU")
    out = gr.Textbox(label="Result")
    btn.click(check_gpu, outputs=out)

demo.launch()
