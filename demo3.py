import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

def analyze(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2%} confident)"

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Textbox(label="Enter some text"),
    outputs=gr.Textbox(label="Sentiment"),
    title="Sentiment Analyzer",
    description="Type a sentence and see if it's positive or negative.",
)

demo.launch()