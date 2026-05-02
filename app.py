from __future__ import annotations

import io
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from dotenv import load_dotenv
from torchvision import models, transforms
from torchvision.models import efficientnet_b0

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - handled in UI
    tf = None
    TF_IMPORT_ERROR = exc
else:
    TF_IMPORT_ERROR = None


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("skin_analysis_app")

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

FACE_PROTO_PATH = APP_DIR / "skintone" / "skinToneCNN" / "deploy.prototxt"
FACE_MODEL_PATH = APP_DIR / "skintone" / "skinToneCNN" / "res10_300x300_ssd_iter_140000.caffemodel"
SKIN_TONE_MODEL_PATH = APP_DIR / "skintone" / "skinToneCNN" / "skintone_mobilenet.pth"
SKIN_TYPE_MODEL_PATH = APP_DIR / "skintype" / "skintypeCNN" / "cnn_skin_model.pth"
SKIN_TYPE_CLASSES_PATH = APP_DIR / "skintype" / "skintypeCNN" / "cnn_classes.pkl"
UNDERTONE_MODEL_PATH = APP_DIR / "undertone" / "undertoneCNN" / "undertone_cnn_best.keras"
UNDERTONE_LABEL_ENCODER_PATH = APP_DIR / "undertone" / "undertoneCNN" / "undertone_label_encoder.pkl"
REQUIREMENTS_PATH = APP_DIR / "requirements.txt"

FACE_INPUT_SIZE = (300, 300)
FACE_CONFIDENCE_THRESHOLD = 0.70
SKIN_TYPE_CLASSES = ["Dry", "Normal", "Oily"]
SKIN_TYPE_IMAGE_SIZE = 224
SKIN_TYPE_MEAN = [0.485, 0.456, 0.406]
SKIN_TYPE_STD = [0.229, 0.224, 0.225]
SKIN_TYPE_HEAD_UNITS = 256
SKIN_TYPE_DROPOUT = 0.30
UNDERTONE_IMAGE_SIZE = 64

<<<<<<< HEAD
DEFAULT_REQUIREMENTS = """streamlit>=1.35,<2.0
=======
DEFAULT_REQUIREMENTS = """streamlit>=-r,<2.0
>>>>>>> 013d08b (Deploy skin analysis app to Hugging Face Space)
numpy>=1.24,<3.0
opencv-python-headless>=4.10
Pillow>=10.0
torch>=2.2
torchvision>=0.17
tensorflow>=2.16
scikit-learn>=1.4
joblib>=1.3
google-generativeai>=0.8
"""


class ModelAssetError(RuntimeError):
    """Raised when a required model artifact is missing or cannot be loaded."""


@dataclass
class PredictionResult:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class AnalysisResult:
    face_confidence: float
    tone: PredictionResult
    skin_type: PredictionResult
    undertone: PredictionResult
    face_image_rgb: np.ndarray
    tone_roi_rgb: np.ndarray
    undertone_roi_rgb: np.ndarray


@dataclass
class ProductRecommendation:
    name: str
    category: str
    ingredients: list[str]
    description: str
    why_it_fits: str
    product_url: str
    image_url: str


@dataclass
class RecommendationPayload:
    intro: str
    cleanse: str
    treat: str
    protect: str
    products: list[ProductRecommendation]
    source: str = "gemini"


class SkinTypeCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        base = efficientnet_b0(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = base.classifier[1].in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, SKIN_TYPE_HEAD_UNITS),
            nn.Hardswish(),
            nn.Dropout(p=SKIN_TYPE_DROPOUT),
            nn.Linear(SKIN_TYPE_HEAD_UNITS, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class ModelOrchestrator:
    _instance: ModelOrchestrator | None = None
    _initialized = False

    def __new__(cls) -> ModelOrchestrator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self.__class__._initialized:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_net = self._load_face_detector()
        self.tone_model, self.tone_idx_map, self.tone_img_size = self._load_skin_tone_model()
        self.skin_type_model, self.skin_type_classes = self._load_skin_type_model()
        self.undertone_model, self.undertone_classes = self._load_undertone_model()

        self.tone_transform = transforms.Compose(
            [
                transforms.Resize((self.tone_img_size, self.tone_img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.skin_type_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((SKIN_TYPE_IMAGE_SIZE, SKIN_TYPE_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(SKIN_TYPE_MEAN, SKIN_TYPE_STD),
            ]
        )
        self.__class__._initialized = True
        LOGGER.info("All models loaded successfully on %s", self.device)

    @staticmethod
    def _ensure_path(path: Path, label: str) -> Path:
        if not path.exists():
            raise ModelAssetError(f"{label} not found at: {path}")
        return path

    def _load_face_detector(self) -> cv2.dnn_Net:
        proto = self._ensure_path(FACE_PROTO_PATH, "Face detector prototxt")
        weights = self._ensure_path(FACE_MODEL_PATH, "Face detector weights")
        try:
            return cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        except cv2.error as exc:
            raise ModelAssetError(f"Unable to load face detector: {exc}") from exc

    def _load_skin_tone_model(self) -> tuple[nn.Module, dict[int, str], int]:
        checkpoint_path = self._ensure_path(SKIN_TONE_MODEL_PATH, "Skin tone model")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            idx_map = checkpoint.get("idx_map", {0: "light", 1: "medium", 2: "dark"})
            img_size = int(checkpoint.get("img_size", 96))

            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[0].in_features
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.35),
                nn.Linear(256, len(idx_map)),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            return model, idx_map, img_size
        except Exception as exc:
            raise ModelAssetError(f"Unable to load skin tone checkpoint: {exc}") from exc

    def _load_skin_type_model(self) -> tuple[nn.Module, list[str]]:
        model_path = self._ensure_path(SKIN_TYPE_MODEL_PATH, "Skin type model")
        classes_path = self._ensure_path(SKIN_TYPE_CLASSES_PATH, "Skin type class mapping")

        try:
            with classes_path.open("rb") as file_obj:
                classes = pickle.load(file_obj)

            model = SkinTypeCNN(num_classes=len(classes))
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            try:
                with classes_path.open("rb") as file_obj:
                    classes = pickle.load(file_obj)
                model = SkinTypeCNN(num_classes=len(classes))
                state_dict = torch.load(model_path, map_location=self.device)
            except Exception as exc:
                raise ModelAssetError(f"Unable to load skin type model: {exc}") from exc
        except Exception as exc:
            raise ModelAssetError(f"Unable to load skin type model: {exc}") from exc

        try:
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            return model, list(classes)
        except Exception as exc:
            raise ModelAssetError(f"Unable to initialize skin type model: {exc}") from exc

    def _load_undertone_model(self) -> tuple[Any, list[str]]:
        self._ensure_path(UNDERTONE_MODEL_PATH, "Undertone model")
        self._ensure_path(UNDERTONE_LABEL_ENCODER_PATH, "Undertone label encoder")

        if tf is None:
            raise ModelAssetError(f"TensorFlow is unavailable: {TF_IMPORT_ERROR}")

        try:
            model = tf.keras.models.load_model(UNDERTONE_MODEL_PATH, compile=False)
            encoder = joblib.load(UNDERTONE_LABEL_ENCODER_PATH)
        except Exception as exc:
            raise ModelAssetError(f"Unable to load undertone model: {exc}") from exc

        return model, list(encoder.classes_)

    def validate_and_crop_face(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        orig_h, orig_w = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, FACE_INPUT_SIZE),
            scalefactor=1.0,
            size=FACE_INPUT_SIZE,
            mean=(104.0, 177.0, 123.0),
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        best_box: tuple[int, int, int, int] | None = None
        best_confidence = 0.0
        padding = 0.08

        for index in range(detections.shape[2]):
            confidence = float(detections[0, 0, index, 2])
            if confidence < best_confidence:
                continue

            x1, y1, x2, y2 = detections[0, 0, index, 3:7]
            left = int(max(0, (x1 - padding) * orig_w))
            top = int(max(0, (y1 - padding) * orig_h))
            right = int(min(orig_w, (x2 + padding) * orig_w))
            bottom = int(min(orig_h, (y2 + padding) * orig_h))

            if right <= left or bottom <= top:
                continue

            best_confidence = confidence
            best_box = (left, top, right, bottom)

        if best_box is None or best_confidence < FACE_CONFIDENCE_THRESHOLD:
            raise ValueError("Face not recognized. Please adjust lighting and retry.")

        left, top, right, bottom = best_box
        face = image_bgr[top:bottom, left:right]
        if face.size == 0:
            raise ValueError("Face not recognized. Please adjust lighting and retry.")

        return face, best_confidence

    @staticmethod
    def _extract_dual_cheek_roi(face_bgr: np.ndarray) -> np.ndarray:
        height, width = face_bgr.shape[:2]
        y1 = int(height * 0.48)
        y2 = int(height * 0.78)
        left = face_bgr[y1:y2, int(width * 0.12) : int(width * 0.40)]
        right = face_bgr[y1:y2, int(width * 0.60) : int(width * 0.88)]
        if left.size == 0 or right.size == 0:
            raise ValueError("Unable to isolate cheek regions from the detected face.")
        return np.concatenate([left, right], axis=1)

    @staticmethod
    def _extract_average_cheek_roi(face_bgr: np.ndarray) -> np.ndarray:
        height, width = face_bgr.shape[:2]
        left = face_bgr[int(height * 0.45) : int(height * 0.75), int(width * 0.05) : int(width * 0.45)]
        right = face_bgr[int(height * 0.45) : int(height * 0.75), int(width * 0.55) : int(width * 0.95)]
        if left.size == 0 or right.size == 0:
            raise ValueError("Unable to isolate cheek regions from the detected face.")
        left_resized = cv2.resize(left, (SKIN_TYPE_IMAGE_SIZE, SKIN_TYPE_IMAGE_SIZE)).astype(np.float32)
        right_resized = cv2.resize(right, (SKIN_TYPE_IMAGE_SIZE, SKIN_TYPE_IMAGE_SIZE)).astype(np.float32)
        return ((left_resized + right_resized) / 2.0).astype(np.uint8)

    def predict_skin_tone(self, tone_roi_bgr: np.ndarray) -> PredictionResult:
        roi_rgb = cv2.cvtColor(tone_roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.tone_transform(Image.fromarray(roi_rgb)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.tone_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        top_index = int(np.argmax(probs))
        label = str(self.tone_idx_map[top_index]).title()
        probability_map = {str(self.tone_idx_map[idx]).title(): float(prob) for idx, prob in enumerate(probs)}
        return PredictionResult(label=label, confidence=float(probs[top_index]), probabilities=probability_map)

    def predict_skin_type(self, face_bgr: np.ndarray) -> PredictionResult:
        roi_rgb = cv2.cvtColor(self._extract_average_cheek_roi(face_bgr), cv2.COLOR_BGR2RGB)
        tensor = self.skin_type_transform(roi_rgb).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.skin_type_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()

        top_index = int(np.argmax(probs))
        label = str(self.skin_type_classes[top_index]).title()
        probability_map = {str(name).title(): float(prob) for name, prob in zip(self.skin_type_classes, probs)}
        return PredictionResult(label=label, confidence=float(probs[top_index]), probabilities=probability_map)

    def predict_undertone(self, undertone_roi_bgr: np.ndarray) -> PredictionResult:
        undertone_lab = cv2.cvtColor(undertone_roi_bgr, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(undertone_lab, (UNDERTONE_IMAGE_SIZE, UNDERTONE_IMAGE_SIZE)).astype(np.float32) / 255.0
        tensor = np.expand_dims(resized, axis=0)
        probs = self.undertone_model.predict(tensor, verbose=0)[0]
        top_index = int(np.argmax(probs))
        label = str(self.undertone_classes[top_index]).title()
        probability_map = {str(name).title(): float(prob) for name, prob in zip(self.undertone_classes, probs)}
        return PredictionResult(label=label, confidence=float(probs[top_index]), probabilities=probability_map)

    def analyze(self, image_bgr: np.ndarray) -> AnalysisResult:
        face_bgr, face_confidence = self.validate_and_crop_face(image_bgr)
        tone_roi_bgr = self._extract_dual_cheek_roi(face_bgr)
        undertone_roi_bgr = tone_roi_bgr.copy()

        tone_result = self.predict_skin_tone(tone_roi_bgr)
        skin_type_result = self.predict_skin_type(face_bgr)
        undertone_result = self.predict_undertone(undertone_roi_bgr)

        return AnalysisResult(
            face_confidence=face_confidence,
            tone=tone_result,
            skin_type=skin_type_result,
            undertone=undertone_result,
            face_image_rgb=cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB),
            tone_roi_rgb=cv2.cvtColor(tone_roi_bgr, cv2.COLOR_BGR2RGB),
            undertone_roi_rgb=cv2.cvtColor(undertone_roi_bgr, cv2.COLOR_BGR2RGB),
        )


def ensure_requirements_file() -> None:
    if REQUIREMENTS_PATH.exists():
        return
    REQUIREMENTS_PATH.write_text(DEFAULT_REQUIREMENTS, encoding="utf-8")
    LOGGER.info("Generated requirements.txt at %s", REQUIREMENTS_PATH)


def get_gemini_api_key() -> str | None:
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key

    try:
        secrets_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        secrets_key = None
    return secrets_key


def build_gemini_prompt(result: AnalysisResult) -> str:
    return (
        "You are a skincare recommendation assistant for a dermatology-style web app. "
        f"The user has {result.tone.label.lower()} skin tone, "
        f"{result.skin_type.label.lower()} skin type, and {result.undertone.label.lower()} undertone. "
        "Explain things in very simple language. "
        "First explain what their undertone and skin type mean and which ingredients suit them and why. "
        "Then create a 3-step routine with Cleanse, Treat, and Protect. "
        "Then recommend exactly 3 products available online. "
        "For each product include: name, category, 2 to 4 highlighted ingredients such as niacinamide 10% or vitamin c 10%, "
        "a short description, why it fits this user, one product URL, and one image URL. "
        "Return only valid JSON using this exact schema: "
        "{"
        "\"intro\":\"simple explanation for the user\","
        "\"routine\":{\"cleanse\":\"...\",\"treat\":\"...\",\"protect\":\"...\"},"
        "\"products\":["
        "{"
        "\"name\":\"...\","
        "\"category\":\"Cleanser or Serum or Sunscreen\","
        "\"ingredients\":[\"...\",\"...\"],"
        "\"description\":\"...\","
        "\"why_it_fits\":\"...\","
        "\"product_url\":\"https://...\","
        "\"image_url\":\"https://...\""
        "}"
        "]"
        "}"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _normalize_recommendation_payload(payload: dict[str, Any]) -> RecommendationPayload:
    routine = payload.get("routine", {})
    products: list[ProductRecommendation] = []
    for item in payload.get("products", [])[:3]:
        products.append(
            ProductRecommendation(
                name=str(item.get("name", "Recommended product")),
                category=str(item.get("category", "Skincare")),
                ingredients=[str(value) for value in item.get("ingredients", []) if str(value).strip()],
                description=str(item.get("description", "")),
                why_it_fits=str(item.get("why_it_fits", "")),
                product_url=str(item.get("product_url", "")).strip(),
                image_url=str(item.get("image_url", "")).strip(),
            )
        )

    return RecommendationPayload(
        intro=str(payload.get("intro", "")).strip(),
        cleanse=str(routine.get("cleanse", "")).strip(),
        treat=str(routine.get("treat", "")).strip(),
        protect=str(routine.get("protect", "")).strip(),
        products=products,
    )


def _friendly_gemini_error(exc: Exception) -> str:
    message = str(exc)
    if "API_KEY_INVALID" in message or "api key not valid" in message.lower():
        return "Your Gemini API key is invalid. Update `GEMINI_API_KEY` in `.env`, restart Streamlit, and try again."
    if "quota" in message.lower():
        return "Your Gemini API quota appears to be exhausted right now. Please check your Google AI quota and retry."
    return f"Gemini recommendation generation failed: {message}"


def build_fallback_recommendations(result: AnalysisResult) -> RecommendationPayload:
    skin_type = result.skin_type.label.lower()
    undertone = result.undertone.label.lower()

    if skin_type == "oily":
        cleanser_name = "CeraVe Foaming Facial Cleanser"
        cleanser_url = "https://www.cerave.com/skincare/cleansers/foaming-facial-cleanser/"
        cleanser_desc = "A gentle foaming cleanser that removes oil and dirt without making the skin feel stripped."
        cleanser_why = "This suits oily or balanced-to-oily skin because it cleans well while still supporting the skin barrier."
        cleanser_ingredients = ["Ceramides", "Niacinamide", "Hyaluronic Acid"]
    else:
        cleanser_name = "CeraVe Hydrating Facial Cleanser"
        cleanser_url = "https://www.cerave.com/en-us/skincare/cleansers/hydrating-facial-cleanser"
        cleanser_desc = "A cream cleanser that washes away dirt while helping skin stay soft and comfortable."
        cleanser_why = "This works well when you want a gentle daily cleanser that supports hydration and the skin barrier."
        cleanser_ingredients = ["Ceramides", "Hyaluronic Acid", "Glycerin"]

    treat_intro = "Niacinamide is a strong match here because it helps with tone balance, barrier support, and a calmer overall look."
    if undertone == "warm":
        treat_intro = "For a warm undertone, brightening and barrier-support ingredients like niacinamide work well to keep skin even and fresh-looking."
    elif undertone == "cool":
        treat_intro = "For a cool undertone, barrier-support and soothing ingredients help the skin look more balanced and less stressed."
    elif undertone == "neutral":
        treat_intro = "For a neutral undertone, balanced ingredients like niacinamide and hydrating support are usually a safe, flexible choice."

    products = [
        ProductRecommendation(
            name=cleanser_name,
            category="Cleanser",
            ingredients=cleanser_ingredients,
            description=cleanser_desc,
            why_it_fits=cleanser_why,
            product_url=cleanser_url,
            image_url="",
        ),
        ProductRecommendation(
            name="The Ordinary Niacinamide 10% + Zinc 1%",
            category="Treat Serum",
            ingredients=["Niacinamide 10%", "Zinc PCA"],
            description="A lightweight serum designed to support brightness, improve texture, and help with visible shine.",
            why_it_fits=f"{treat_intro} It is especially useful when you want a simple treatment step without making the routine too heavy.",
            product_url="https://theordinary.com/en-us/niacinamide-10-zinc-1-serum-100436.html",
            image_url="",
        ),
        ProductRecommendation(
            name="CeraVe AM Facial Moisturizing Lotion SPF 50",
            category="Protect",
            ingredients=["SPF 50", "Niacinamide", "Ceramides", "Hyaluronic Acid"],
            description="A daytime moisturizer with broad-spectrum SPF 50 that hydrates while protecting skin from sun exposure.",
            why_it_fits="Daily sun protection matters for every undertone and skin tone because it helps prevent dullness, uneven tone, and barrier stress.",
            product_url="https://www.cerave.com/skincare/moisturizers/facial-moisturizers/am-facial-moisturizing-lotion-spf-50",
            image_url="",
        ),
    ]

    intro = (
        f"Your skin reads as {result.skin_type.label.lower()} with a {result.undertone.label.lower()} undertone. "
        "That usually means your routine should focus on keeping the skin barrier steady, avoiding harsh stripping products, "
        "and using ingredients that support brightness and calm. "
        "Niacinamide is a good example because it can help the skin look more balanced, support the barrier, and reduce the look of excess oil or uneven texture."
    )

    cleanse = "Use a gentle cleanser morning and night. It should remove dirt, sunscreen, and extra oil without leaving your face feeling tight."
    treat = "Use a niacinamide-based serum after cleansing. This can help with balance, visible shine, texture, and overall skin comfort."
    protect = "Finish with a daily SPF product every morning. Sunscreen helps prevent tanning, uneven tone, and long-term skin stress from UV exposure."

    return RecommendationPayload(
        intro=intro,
        cleanse=cleanse,
        treat=treat,
        protect=protect,
        products=products,
        source="fallback",
    )


def generate_skincare_recommendations(result: AnalysisResult) -> RecommendationPayload:
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key missing. Set `GEMINI_API_KEY` in your environment or Streamlit secrets."
        )

    try:
        import google.generativeai as genai
    except Exception as exc:
        raise RuntimeError(f"`google-generativeai` is not installed or failed to import: {exc}") from exc

    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(build_gemini_prompt(result))
        text = getattr(response, "text", "") or ""
    except Exception as exc:
        raise RuntimeError(_friendly_gemini_error(exc)) from exc

    if not text.strip():
        raise RuntimeError("Gemini returned an empty recommendation response.")
    try:
        return _normalize_recommendation_payload(_extract_json_object(text))
    except Exception as exc:
        raise RuntimeError(f"Gemini returned an invalid recommendation format: {exc}") from exc


@st.cache_resource(show_spinner=False)
def get_orchestrator() -> ModelOrchestrator:
    return ModelOrchestrator()


def bytes_to_bgr_image(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The selected file could not be decoded as an image.")
    return image


def render_metric_card(title: str, prediction: PredictionResult) -> None:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="card-label">{title}</div>
            <div class="card-value">{prediction.label}</div>
            <div class="card-meta">Confidence: {prediction.confidence:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_table(title: str, probabilities: dict[str, float]) -> None:
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    with st.container(border=False):
        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="confidence-title">{title} confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for label, score in ordered:
            st.caption(f"{label}: {score:.1%}")
            st.progress(float(score))


def render_input_mode_selector() -> str:
    current_mode = st.session_state.get("input_mode", "Upload Image")
    upload_col, capture_col = st.columns(2)

    with upload_col:
        upload_selected = current_mode == "Upload Image"
        st.markdown(
            f"""
            <div class="selector-card {'selected' if upload_selected else ''}">
                <div class="selector-bubble">{'●' if upload_selected else '○'}</div>
                <div class="selector-title">Upload</div>
                <div class="selector-copy">Choose a clear image from your device.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Use Upload", key="upload_mode_button", use_container_width=True):
            st.session_state["input_mode"] = "Upload Image"

    with capture_col:
        capture_selected = current_mode == "Capture Image"
        st.markdown(
            f"""
            <div class="selector-card {'selected' if capture_selected else ''}">
                <div class="selector-bubble">{'●' if capture_selected else '○'}</div>
                <div class="selector-title">Capture</div>
                <div class="selector-copy">Take a live, front-facing photo with good light.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Use Capture", key="capture_mode_button", use_container_width=True):
            st.session_state["input_mode"] = "Capture Image"

    return st.session_state.get("input_mode", current_mode)


<<<<<<< HEAD
=======
def _store_selected_image(payload: bytes, source: str) -> tuple[np.ndarray, Image.Image]:
    st.session_state["selected_image_bytes"] = payload
    st.session_state["selected_image_source"] = source
    st.session_state["image_ready"] = True
    return bytes_to_bgr_image(payload), Image.open(io.BytesIO(payload)).convert("RGB")


def _restore_selected_image(expected_source: str) -> tuple[np.ndarray | None, Image.Image | None]:
    payload = st.session_state.get("selected_image_bytes")
    source = st.session_state.get("selected_image_source")
    if not payload or source != expected_source:
        return None, None
    try:
        return bytes_to_bgr_image(payload), Image.open(io.BytesIO(payload)).convert("RGB")
    except Exception:
        return None, None


>>>>>>> 013d08b (Deploy skin analysis app to Hugging Face Space)
def render_user_summary(analysis: AnalysisResult) -> None:
    message = (
        f"Your skin profile looks {analysis.skin_type.label.lower()} with a {analysis.undertone.label.lower()} undertone. "
        "The best routine for you is gentle, barrier-friendly, and consistent. "
        "Look for smart actives like niacinamide, daily hydration, and sunscreen every morning."
    )
    st.markdown(
        f"""
        <div class="explain-card">
            <div class="explain-title">What suits your skin best</div>
            <div class="explain-copy">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation_payload(payload: RecommendationPayload) -> None:
    if payload.intro:
        st.markdown(
            f"""
            <div class="explain-card">
                <div class="explain-title">Your personalized skincare direction</div>
                <div class="explain-copy">{payload.intro}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    routine_cols = st.columns(3)
    routine_items = [
        ("Cleanse", payload.cleanse),
        ("Treat", payload.treat),
        ("Protect", payload.protect),
    ]
    for column, (title, description) in zip(routine_cols, routine_items):
        with column:
            st.markdown(
                f"""
                <div class="routine-card">
                    <div class="routine-label">{title}</div>
                    <div class="routine-copy">{description or 'Recommendation unavailable.'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("### Best product picks for you")
    for index, product in enumerate(payload.products, start=1):
        st.markdown(
            f"""
            <div class="product-section-label">Recommendation {index}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="product-feature-card">
                <div class="product-badge">Best for you</div>
                <div class="product-category">{product.category}</div>
                <div class="product-name">{product.name}</div>
                <div class="product-copy">{product.description}</div>
                <div class="product-why">{product.why_it_fits}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if product.ingredients:
            chips = "".join(f'<span class="ingredient-pill">{ingredient}</span>' for ingredient in product.ingredients)
            st.markdown(chips, unsafe_allow_html=True)
        if product.product_url:
            st.link_button("Open product link", product.product_url, use_container_width=False)
        st.markdown('<div class="product-divider"></div>', unsafe_allow_html=True)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ink-900: #143447;
                --ink-700: #36596d;
                --ink-500: #5f7c8f;
                --line-soft: rgba(79, 136, 166, 0.16);
                --panel-white: rgba(255, 255, 255, 0.94);
                --shadow-soft: 0 12px 30px rgba(38, 97, 124, 0.08);
            }
            .stApp {
                background: linear-gradient(180deg, #f4fafe 0%, #eef7fb 50%, #ffffff 100%);
                color: var(--ink-900);
            }
            .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
                color: var(--ink-700);
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
                color: var(--ink-900);
                letter-spacing: -0.02em;
            }
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 3rem;
            }
            .hero-shell {
                padding: 1.6rem 1.8rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(18, 87, 123, 0.96), rgba(99, 170, 201, 0.88));
                color: #f8fdff;
                box-shadow: 0 18px 40px rgba(43, 103, 132, 0.18);
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }
            .hero-shell::after {
                content: "";
                position: absolute;
                top: -80px;
                right: -40px;
                width: 220px;
                height: 220px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 70%);
            }
            .hero-eyebrow {
                display: inline-block;
                padding: 0.32rem 0.7rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.14);
                color: #f4fbff !important;
                font-size: 0.78rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 0.9rem;
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
                color: #f8fdff !important;
            }
            .hero-copy {
                font-size: 1rem;
                opacity: 0.95;
                margin: 0;
                color: #eefafe !important;
            }
            .hero-chip-row {
                display: flex;
                gap: 0.65rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            }
            .hero-chip {
                padding: 0.38rem 0.78rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.14);
                border: 1px solid rgba(255,255,255,0.14);
                color: #f7fdff !important;
                font-size: 0.86rem;
                font-weight: 600;
            }
            .section-shell {
                display: flex;
                align-items: center;
                gap: 0.95rem;
                margin: 1.6rem 0 0.95rem 0;
                padding-bottom: 0.85rem;
                border-bottom: 1px solid rgba(79,136,166,0.16);
            }
            .section-step {
                width: 42px;
                height: 42px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #347fa4 0%, #8acbbb 100%);
                color: #ffffff !important;
                font-size: 1rem;
                font-weight: 700;
                flex-shrink: 0;
                box-shadow: 0 8px 18px rgba(58, 120, 150, 0.18);
            }
            .section-text {
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
            }
            .section-title {
                color: var(--ink-900) !important;
                font-size: 1.45rem;
                font-weight: 700;
                line-height: 1.1;
            }
            .section-subtitle {
                color: var(--ink-500) !important;
                font-size: 0.95rem;
                line-height: 1.45;
            }
            .result-card {
                background: var(--panel-white);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: var(--shadow-soft);
                min-height: 145px;
            }
            .card-label {
                color: #5b8295;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.77rem;
                font-weight: 700;
                margin-bottom: 0.6rem;
            }
            .card-value {
                color: var(--ink-900);
                font-size: 1.45rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .card-meta {
                color: var(--ink-500);
                font-size: 0.92rem;
            }
            .confidence-card {
                margin-top: 0.7rem;
            }
            .confidence-title {
                color: var(--ink-700);
                font-size: 0.86rem;
                font-weight: 700;
                margin-bottom: 0.3rem;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }
            .panel {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
                padding: 1rem;
            }
            .selector-card {
                background: var(--panel-white);
                border: 1px solid rgba(79, 136, 166, 0.18);
                border-radius: 22px;
                padding: 1.1rem;
                text-align: center;
                min-height: 160px;
                box-shadow: 0 10px 24px rgba(38, 97, 124, 0.06);
                margin-bottom: 0.5rem;
            }
            .selector-card.selected {
                border: 2px solid #4f9fc4;
                box-shadow: 0 12px 28px rgba(79, 159, 196, 0.18);
                background: linear-gradient(180deg, #fafdff 0%, #f0f8fc 100%);
            }
            .selector-bubble {
                font-size: 2rem;
                color: #4f9fc4;
                line-height: 1;
                margin-bottom: 0.7rem;
            }
            .selector-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: var(--ink-900);
                margin-bottom: 0.35rem;
            }
            .selector-copy {
                color: var(--ink-500);
                font-size: 0.95rem;
            }
            .explain-card, .routine-card, .product-card {
                color: #24485d;
                background: var(--panel-white);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 10px 26px rgba(38, 97, 124, 0.06);
            }
            .explain-card {
                margin: 1rem 0 1.2rem 0;
            }
            .explain-title, .routine-label, .product-name {
                color: var(--ink-900) !important;
                font-weight: 700;
            }
            .explain-title, .product-name {
                font-size: 1.1rem;
                margin-bottom: 0.4rem;
            }
            .explain-copy, .routine-copy, .product-copy, .product-why {
                color: var(--ink-700) !important;
                font-size: 0.97rem;
                line-height: 1.6;
            }
            .routine-card {
                min-height: 170px;
                margin-bottom: 1rem;
            }
            .routine-label {
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.82rem;
                margin-bottom: 0.7rem;
            }
            .product-category {
                color: #5b8295;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.76rem;
                font-weight: 700;
                margin-bottom: 0.4rem;
            }
            .product-badge {
                display: inline-block;
                margin-bottom: 0.65rem;
                padding: 0.28rem 0.6rem;
                border-radius: 999px;
                background: linear-gradient(90deg, #dff4ff 0%, #eef8dd 100%);
                color: #1d5c79;
                font-size: 0.76rem;
                font-weight: 700;
            }
            .ingredient-pill {
                display: inline-block;
                background: #edf7fb;
                color: #1d5c79;
                border: 1px solid rgba(79, 136, 166, 0.18);
                padding: 0.4rem 0.65rem;
                margin: 0.2rem 0.35rem 0.5rem 0;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 600;
            }
            .product-feature-card {
                background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(247,252,255,0.96) 100%);
                border: 1px solid rgba(79, 136, 166, 0.16);
                border-radius: 20px;
                padding: 1.15rem 1.2rem;
                box-shadow: 0 14px 28px rgba(38, 97, 124, 0.08);
                margin-bottom: 0.6rem;
                position: relative;
                overflow: hidden;
            }
            .product-feature-card::before {
                content: "";
                position: absolute;
                top: -48px;
                right: -28px;
                width: 120px;
                height: 120px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(117, 196, 202, 0.18) 0%, rgba(117, 196, 202, 0) 72%);
            }
            .product-section-label {
                color: var(--ink-500);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.8rem;
                font-weight: 700;
                margin: 0.8rem 0 0.6rem 0;
            }
            .product-divider {
                height: 1px;
                background: linear-gradient(90deg, transparent 0%, rgba(79,136,166,0.22) 50%, transparent 100%);
                margin: 1.2rem 0 0.6rem 0;
            }
            [data-testid="stExpander"] {
                background: rgba(255,255,255,0.74);
                border: 1px solid var(--line-soft);
                border-radius: 18px;
            }
            [data-testid="stExpander"] summary p {
                color: var(--ink-700) !important;
                font-weight: 600;
            }
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #4a96b8 0%, #8ed0c1 100%) !important;
            }
            .stCaption {
                color: var(--ink-500) !important;
            }
            .stButton button, .stLinkButton a {
                border-radius: 999px !important;
                font-weight: 600 !important;
            }
            .stLinkButton a {
                background: #16384c !important;
                color: #ffffff !important;
                border: none !important;
                padding: 0.5rem 1rem !important;
            }
            @media (max-width: 768px) {
                .block-container {
                    padding-top: 1rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
                .hero-shell {
                    padding: 1.2rem 1rem;
                    border-radius: 18px;
                }
                .hero-title {
                    font-size: 1.55rem;
                }
                .hero-chip-row {
                    gap: 0.45rem;
                }
                .hero-chip {
                    font-size: 0.8rem;
                }
                .section-shell {
                    gap: 0.75rem;
                    margin-top: 1.2rem;
                }
                .section-step {
                    width: 38px;
                    height: 38px;
                    font-size: 0.95rem;
                }
                .section-title {
                    font-size: 1.2rem;
                }
                .section-subtitle {
                    font-size: 0.88rem;
                }
                .result-card, .product-feature-card, .explain-card, .routine-card {
                    border-radius: 16px;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-eyebrow">AI Skin Studio</div>
            <div class="hero-title">Dermatology Clinic Skin Analysis</div>
            <p class="hero-copy">
                AI-assisted skin tone, skin type, and undertone profiling with personalized skincare guidance.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">Face-validated</span>
                <span class="hero-chip">Personalized routine</span>
                <span class="hero-chip">Product picks</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(step: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-step">{step}</div>
            <div class="section-text">
                <div class="section-title">{title}</div>
                <div class="section-subtitle">{subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_input_image() -> tuple[np.ndarray | None, Image.Image | None]:
    input_mode = render_input_mode_selector()
    st.caption("Use a clear, front-facing photo in soft daylight for the most accurate analysis.")

    if input_mode == "Upload Image":
<<<<<<< HEAD
        uploaded = st.file_uploader("Upload a facial image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is None:
            return None, None
        payload = uploaded.getvalue()
        return bytes_to_bgr_image(payload), Image.open(io.BytesIO(payload)).convert("RGB")

    captured = st.camera_input("Capture a live image")
    if captured is None:
        return None, None
    payload = captured.getvalue()
    return bytes_to_bgr_image(payload), Image.open(io.BytesIO(payload)).convert("RGB")
=======
        uploaded = st.file_uploader(
            "Upload a facial image",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_input",
            help="JPG, PNG, or WEBP. Front-facing images work best.",
        )
        if uploaded is not None:
            payload = uploaded.getvalue()
            return _store_selected_image(payload, "upload")
        return _restore_selected_image("upload")

    try:
        captured = st.camera_input("Capture a live image", key="camera_input")
    except Exception as exc:
        st.warning(f"Camera access is currently unavailable in this browser session: {exc}")
        return _restore_selected_image("camera")

    if captured is not None:
        payload = captured.getvalue()
        return _store_selected_image(payload, "camera")
    return _restore_selected_image("camera")
>>>>>>> 013d08b (Deploy skin analysis app to Hugging Face Space)


def render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Deployment Checks")
        st.caption(f"Requirements file: `{REQUIREMENTS_PATH.name}`")
        st.caption(f"Device: `{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}`")
        st.caption("Gemini env var: `GEMINI_API_KEY`")
        with st.expander("Model inventory", expanded=False):
            st.code(
                "\n".join(
                    [
                        str(SKIN_TONE_MODEL_PATH.relative_to(APP_DIR)),
                        str(SKIN_TYPE_MODEL_PATH.relative_to(APP_DIR)),
                        str(UNDERTONE_MODEL_PATH.relative_to(APP_DIR)),
                        str(FACE_MODEL_PATH.relative_to(APP_DIR)),
                    ]
                ),
                language="text",
            )


def main() -> None:
    st.set_page_config(page_title="Skin Analysis Clinic", layout="wide")
    ensure_requirements_file()
    inject_styles()
    render_sidebar()
    render_header()

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    input_bgr, preview_image = load_input_image()
    if preview_image is not None and st.session_state.get("image_ready", True):
        st.image(preview_image, caption="Selected image", use_container_width=True)
<<<<<<< HEAD
=======
        clear_col, analyze_hint_col = st.columns([1, 3], vertical_alignment="center")
        with clear_col:
            if st.button("Clear photo", key="clear_photo_button", use_container_width=True):
                st.session_state.pop("selected_image_bytes", None)
                st.session_state.pop("selected_image_source", None)
                st.session_state["image_ready"] = False
        with analyze_hint_col:
            st.caption("Image ready. Click the analyze button below to start the skin assessment.")
>>>>>>> 013d08b (Deploy skin analysis app to Hugging Face Space)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.get("image_ready") is False:
        st.info("Upload or capture a clear, front-facing image to begin the analysis.")
        return

    if input_bgr is None:
        st.info("Upload or capture a clear, front-facing image to begin the analysis.")
        return

    if not st.button("Analyze Skin Profile", type="primary", use_container_width=True):
        return

    try:
        orchestrator = get_orchestrator()
    except ModelAssetError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - UI guardrail
        LOGGER.exception("Unexpected model loading failure")
        st.error(f"Unexpected model loading failure: {exc}")
        st.stop()

    try:
        with st.spinner("Analyzing Skin Bio-markers..."):
            analysis = orchestrator.analyze(input_bgr)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except ModelAssetError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:  # pragma: no cover - UI guardrail
        LOGGER.exception("Inference failed")
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.success(f"Face validated with {analysis.face_confidence:.1%} confidence.")

    render_section_heading("1", "Your skin analysis", "A quick summary of your detected tone, type, and undertone.")
    tone_col, type_col, undertone_col = st.columns(3)
    with tone_col:
        render_metric_card("Skin Tone", analysis.tone)
        render_probability_table("Skin tone", analysis.tone.probabilities)
    with type_col:
        render_metric_card("Skin Type", analysis.skin_type)
        render_probability_table("Skin type", analysis.skin_type.probabilities)
    with undertone_col:
        render_metric_card("Undertone", analysis.undertone)
        render_probability_table("Undertone", analysis.undertone.probabilities)

    with st.expander("View extracted analysis regions", expanded=False):
        face_col, cheek_col = st.columns(2)
        with face_col:
            st.image(analysis.face_image_rgb, caption="Detected face crop", use_container_width=True)
        with cheek_col:
            st.image(analysis.tone_roi_rgb, caption="Cheek ROI used for tone and undertone", use_container_width=True)

    render_section_heading("2", "What this means", "Short, practical guidance based on your skin profile.")
    render_user_summary(analysis)
    render_section_heading("3", "Your skincare routine", "A simple daily plan with product picks chosen for your profile.")
    try:
        payload = generate_skincare_recommendations(analysis)
        render_recommendation_payload(payload)
    except RuntimeError as exc:
        payload = build_fallback_recommendations(analysis)
        render_recommendation_payload(payload)


if __name__ == "__main__":
    main()
