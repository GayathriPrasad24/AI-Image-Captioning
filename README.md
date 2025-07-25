---
title: Image Captioning
emoji: 🖼️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 5.38.2
app_file: app.py
pinned: false
---

# 🖼️ Image Captioning

Generate natural language captions for any image using the `vit-gpt2` transformer model — right from your browser!

[![Gradio](https://img.shields.io/badge/Made%20with-Gradio-FF6E42?logo=gradio)](https://gradio.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/GayathriPrasad24/AI-Image-Captioning)

---

## 🚀 How to Use

1. **Upload or drag-and-drop** an image (JPG, PNG).
2. Click **"Generate Caption"**.
3. See the **generated caption**, download it, and view it in the gallery.

---

## 🔍 Model Used

- [`nlpconnect/vit-gpt2-image-captioning`](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)
- Combines **Vision Transformer (ViT)** and **GPT2** to convert images into human-like captions.

---

## ✨ Features

- 📂 Drag-and-drop image upload
- 🧠 AI-generated captions using Transformers
- 💾 Download caption as `.txt`
- 🖼️ Image + caption gallery
- ☁️ Hosted on Hugging Face Spaces

---

## 🛠 Tech Stack

- `Transformers` (Hugging Face)
- `Torch` (PyTorch)
- `Gradio` (Frontend UI)
- `Pillow`, `NetworkX`, etc.

---

## 📁 Repository

🔗 GitHub: [GayathriPrasad24/AI-Image-Captioning](https://github.com/GayathriPrasad24/AI-Image-Captioning)

---

## 🖥️ Run Locally

```bash
git clone https://github.com/GayathriPrasad24/AI-Image-Captioning.git
cd AI-Image-Captioning
pip install -r requirements.txt
python app.py
```

---

> ✨ Built with ❤️ by [GayathriPrasad24](https://huggingface.co/GayathriPrasad24)
