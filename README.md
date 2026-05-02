---
title: AI Skin Analysis App
emoji: 🧴
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Skin Analysis Web App

A production-oriented Streamlit application for AI-assisted skin analysis and skincare recommendations.

This project combines four selected models into one deployable experience:

- Face Detection: SSD ResNet
- Skin Tone: MobileNet
- Skin Type: CNN
- Undertone: CNN

The app analyzes a face image, predicts tone/type/undertone, and generates a simple skincare routine with product recommendations.

## Features

- Upload image or capture image from camera
- Face validation before inference
- Unified model loading through a singleton `ModelOrchestrator`
- Skin tone, skin type, and undertone prediction in one flow
- Gemini-powered skincare recommendations
- Automatic fallback recommendations if Gemini is unavailable
- Modern clinic-style Streamlit UI
- Deployment-ready environment and requirements setup

## Models Used

Only these models are integrated in the web app:

- Face detector: `skintone/skinToneCNN/res10_300x300_ssd_iter_140000.caffemodel`
- Face detector config: `skintone/skinToneCNN/deploy.prototxt`
- Skin tone model: `skintone/skinToneCNN/skintone_mobilenet.pth`
- Skin type model: `skintype/skintypeCNN/cnn_skin_model.pth`
- Skin type classes: `skintype/skintypeCNN/cnn_classes.pkl`
- Undertone model: `undertone/undertoneCNN/undertone_cnn_best.keras`
- Undertone label encoder: `undertone/undertoneCNN/undertone_label_encoder.pkl`

## Tech Stack

- Python
- Streamlit
- PyTorch
- TensorFlow / Keras
- OpenCV
- Google Gemini API

## Project Structure

```text
miniProject/
├── app.py
├── requirements.txt
├── .env.example
├── SETUP_COMMANDS.md
├── skintone/
│   └── skinToneCNN/
│       ├── deploy.prototxt
│       ├── res10_300x300_ssd_iter_140000.caffemodel
│       └── skintone_mobilenet.pth
├── skintype/
│   └── skintypeCNN/
│       ├── cnn_skin_model.pth
│       └── cnn_classes.pkl
└── undertone/
    └── undertoneCNN/
        ├── undertone_cnn_best.keras
        └── undertone_label_encoder.pkl
```

## How It Works

1. The user uploads or captures a face image.
2. The SSD face detector validates that a face is present.
3. The app crops the face and prepares model-specific regions.
4. The three prediction models return:
   - Skin tone
   - Skin type
   - Undertone
5. Gemini generates a concise skincare routine and product suggestions.
6. If Gemini is unavailable, the app shows built-in fallback recommendations.

## Local Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd miniProject
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.11 is not available, use Python 3.12.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file from `.env.example` and add your Gemini key:

```env
GEMINI_API_KEY=your_real_api_key_here
```

Optional:

```env
GEMINI_MODEL=gemini-1.5-flash
```

### 5. Run the app

```powershell
streamlit run app.py
```

If `streamlit` is not recognized:

```powershell
python -m streamlit run app.py
```

## Requirements

- Python 3.11 or 3.12 recommended
- A valid Gemini API key for live LLM recommendations
- Local model files present in the paths listed above

## Environment Variables

Supported environment variables:

- `GEMINI_API_KEY`


## UI Overview

The Streamlit app is organized into three steps:

- `1. Your skin analysis`
- `2. What this means`
- `3. Your skincare routine`

The recommendation section is designed to stay useful even if the LLM is unavailable.

## Deployment Notes

This project is suitable for:

- Local demo use
- GitHub version control
- Hugging Face Spaces deployment

Before deployment:

- Do not commit `.env`
- Keep `.env.example` with placeholders only
- Make sure all required model files are included or available in deployment storage
- Add the Gemini API key as a secret in the deployment platform

## Security Notes

- `.env` should remain private
- API keys must never be hardcoded into `app.py`
- Use deployment secrets for production hosting

<<<<<<< HEAD
=======
## Current Status

- Unified Streamlit app implemented
- UI polished for presentation and public demo
- Fallback recommendation flow added
- GitHub-ready repository structure prepared

## Next Recommended Steps

- Push the project to GitHub
- Deploy to Hugging Face Spaces
- Add platform secrets for Gemini
- Test the hosted inference flow end to end

## License

Add your preferred license here.

>>>>>>> 013d08b (Deploy skin analysis app to Hugging Face Space)
