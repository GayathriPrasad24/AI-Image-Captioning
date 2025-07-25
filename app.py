import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import uuid
import os

# Load model & processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Store history
gallery_items = []

# Caption generation function
def generate_caption(image):
    if image is None:
        return "No image provided", None, gallery_items

    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Save caption to .txt file for download
    filename = f"caption_{uuid.uuid4().hex[:8]}.txt"
    with open(filename, "w") as f:
        f.write(caption)

    # Append to gallery
    gallery_items.insert(0, (image, caption))

    return caption, filename, gallery_items

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Image Captioning App\nUpload or drag & drop an image to get a caption.")

    with gr.Row():
        image_input = gr.Image(type="pil", sources=["upload"], label="Upload Image")
        caption_output = gr.Textbox(label="Generated Caption", lines=2)
    
    with gr.Row():
        caption_button = gr.Button("📸 Generate Caption")
        download_button = gr.File(label="📥 Download Caption")

    gallery = gr.Gallery(label="🖼️ Image + Caption Gallery", columns=3, rows=2)

    caption_button.click(
        fn=generate_caption,
        inputs=image_input,
        outputs=[caption_output, download_button, gallery]
    )

# Launch the app
demo.launch()
