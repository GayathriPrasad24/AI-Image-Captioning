# 🖼️ Image Captioning

Generate **aesthetic, poetic, mood-based captions** for any image using AI — right from your browser.

[![Gradio](https://img.shields.io/badge/Made%20with-Gradio-FF6E42?logo=gradio&style=for-the-badge)](https://gradio.app/)
[![GitHub](https://img.shields.io/badge/View%20on-GitHub-181717?logo=github&style=for-the-badge)](https://github.com/GayathriPrasad24/AI-Image-Captioning)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A5%BC%20Live%20Demo-HuggingFace-blue?logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/GayathriPrasad24/image-captioning)

---

## 🚀 How to Use

1. **Upload**, drag & drop, or use **webcam** to submit an image.
2. Click **"Generate Caption"**.
3. View a beautiful, mood-driven caption — no generic object description!

---

## ✨ Features

- 📸 Webcam, Upload, or Drag-and-Drop Image Support  
- 🎨 Generates *aesthetic, poetic* captions — not just object labels  
- ⚡ Fast, browser-based generation (no sign-in required)  
- ☁️ Hosted on Hugging Face Spaces  
- 🌈 Modern, attractive Gradio UI  

---

## 🔍 Model Used

- [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Combines **Vision Transformer (ViT)** and **language decoder** to produce expressive image descriptions.

---

## 🛠 Tech Stack

- 🤗 `transformers`
- 🔥 `torch` (PyTorch)
- 🎛️ `gradio`
- 🖼️ `Pillow`

---

## 🖼️ Demo

Check out the live demo hosted on Hugging Face Spaces:  
👉 [Hugging Face Space](https://huggingface.co/spaces/GayathriPrasad24/image-captioning)

---

## 📁 Run Locally

```bash
git clone https://github.com/GayathriPrasad24/AI-Image-Captioning.git
cd AI-Image-Captioning
pip install -r requirements.txt
python app.py
