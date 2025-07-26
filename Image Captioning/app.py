import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load the BLIP model for aesthetic captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Poetic touch function
def make_it_poetic(caption):
    poetic_starters = [
        "bathed in golden light,",
        "a soft whisper of dreams,",
        "as twilight kisses the sky,",
        "lost in a quiet reverie,",
        "echoes of forgotten wonder,"
    ]
    starter = poetic_starters[torch.randint(0, len(poetic_starters), (1,)).item()]
    return f"{starter} {caption.strip().capitalize()}."

# Caption generation
def generate_caption(image):
    if image is None:
        return "⚠️ Please upload an image."
    
    try:
        inputs = processor(image, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
        raw_caption = processor.decode(output[0], skip_special_tokens=True)
        return make_it_poetic(raw_caption)
    except Exception as e:
        return f"❌ Error generating caption: {str(e)}"

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌸 Aesthetic Image Captioning")
    gr.Markdown("_Upload an image or use webcam to generate poetic, emotion-based captions._")

    image_input = gr.Image(type="pil", sources=["upload", "webcam"], label=None, show_label=False, visible=True)
    generate_btn = gr.Button("✨ Generate Poetic Caption")
    output_caption = gr.Textbox(label="📝 Caption", placeholder="Your poetic caption will appear here...")

    generate_btn.click(fn=generate_caption, inputs=image_input, outputs=output_caption)

# Launch
if __name__ == "__main__":
    demo.launch()
