import gradio as gr
from PIL import Image
from pipeline import HitDetectorPipeline

pipe = HitDetectorPipeline("model.pt")

def detect(image: Image.Image):
    return pipe(image)

gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Hit Detector").launch(server_name="0.0.0.0", server_port=7860)
