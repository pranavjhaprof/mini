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

## 🚀 Live Demo

**Try the app now:** [AI Skin Analysis App on Hugging Face Spaces](https://pranav2803-ai-skin-analysis-app.hf.space/)

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

## Deployment Instructions

### Hugging Face Spaces

The app is already deployed to [Hugging Face Spaces](https://pranav2803-ai-skin-analysis-app.hf.space/) with Docker.

**To deploy your own version:**

1. Fork this repository or create a new Hugging Face Space
2. Connect your GitHub repository to Hugging Face Spaces
3. Set the SDK to Docker in the Space settings
4. Add the following secrets in Space settings:
   - `GEMINI_API_KEY`: Your Gemini API key
5. Hugging Face will automatically build and deploy from the `hf-deploy` branch
6. Push updates to the `hf-deploy` branch to trigger automatic redeployment

**Before deploying:**

- Ensure `.env` is in `.gitignore` (never commit secrets)
- Keep `.env.example` with placeholders only
- Verify all required model files are in the repository
- Add the Gemini API key as a Space secret, not in the code
- Test locally with `streamlit run app.py` before pushing

## Security Notes

- `.env` should remain private and added to `.gitignore`
- API keys must never be hardcoded into `app.py`
- Always use deployment secrets (Space Secrets on Hugging Face) for production hosting
- Rotate API keys regularly
- Do not share API keys in issues or pull requests

## Current Status

✅ **Deployed to production** on Hugging Face Spaces  
✅ Unified Streamlit app implemented  
✅ UI polished for presentation and public demo  
✅ Fallback recommendation flow added  
✅ GitHub-ready repository structure prepared  
✅ Docker containerization complete  
✅ Automated deployment pipeline configured  

## Future Enhancements

- Add sample images or demo GIF to README
- Document model training notebooks
- Implement analysis history/save feature
- Support more skin classes and broader datasets
- Add PDF/report export functionality
- Implement user feedback loop for model improvement

## License

MIT License - feel free to use this project for educational and commercial purposes. See LICENSE file for details.
