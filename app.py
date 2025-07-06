
import gradio as gr
from transformers import pipeline

# Load a text generation pipeline
generator = pipeline("text-generation", model="gpt2")

def generate_text(prompt):
    result = generator(prompt, max_length=50, num_return_sequences=1)
    return result[0]["generated_text"]

# Gradio interface
demo = gr.Interface(fn=generate_text, inputs="text", outputs="text", title="Text Generator")
demo.launch()
