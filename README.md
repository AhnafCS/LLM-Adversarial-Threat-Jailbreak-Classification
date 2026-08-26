# 🛡️ LLM Adversarial Threat Classifier

Hi! This is our lab project for CSE440 (Natural Language Processing II). 

We built a machine learning pipeline that catches bad prompts (like jailbreaks or toxic content) before they reach an LLM. We fine-tuned a BERT model that categorizes these threats into 5 different classes. It even supports extracting text from PDFs and images using OCR! 🚀

## ✨ Features
* **Real-time scanning:** Instantly checks prompts and shows confidence scores.
* **File Uploads:** Drag and drop PDFs or images to scan text directly.
* **High Accuracy:** We hit a Macro F1 score of **96.64%**!

## 🛠️ How to run locally

1. Clone the repo:
   ```bash
   git clone https://github.com/AhnafCS/LLM-Adversarial-Threat-Jailbreak-Classification.git
   cd LLM-Adversarial-Threat-Jailbreak-Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You'll also need [Tesseract](https://github.com/tesseract-ocr/tesseract) installed if you want to test the image OCR feature)*

3. Run the app:
   ```bash
   streamlit run app_bert.py
   ```

## 🌐 Deployment
This codebase is fully ready to be deployed on **Streamlit Community Cloud**. Just connect the repo and let Streamlit handle the rest! 

---
**Course:** CSE440 (Summer 2026) 🎓
