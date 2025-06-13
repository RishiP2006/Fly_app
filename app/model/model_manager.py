import os
from pathlib import Path
import cv2
import numpy as np
from typing import Dict, Any, List

# YOLO import
from ultralytics import YOLO

# Keras import
import tensorflow as tf
from tensorflow.keras.models import load_model

# Directory paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"

# Info about models (lazy): model_name -> {"path": Path, "type": "yolo" or "keras"}
models_info: Dict[str, Dict[str, Any]] = {}
# Loaded model instances: model_name -> loaded model object (YOLO instance or tf.keras.Model)
loaded_models: Dict[str, Any] = {}


def scan_models():
    """
    Scan MODEL_DIR for supported model files (.pt for YOLO, .keras/.h5 for Keras).
    Populate models_info dict with entries: model_name -> {"path": Path, "type": ...}.
    """
    global models_info
    models_info = {}
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found: {MODEL_DIR}")
    for file_path in MODEL_DIR.iterdir():
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix == ".pt":
            model_type = "yolo"
        elif suffix in (".keras", ".h5"):
            model_type = "keras"
        else:
            continue
        name = file_path.stem  # filename without extension
        models_info[name] = {"path": file_path, "type": model_type}
    if not models_info:
        print(f"[model_manager] WARNING: No models found in {MODEL_DIR}")
    else:
        print(f"[model_manager] Found models: {list(models_info.keys())}")


def available_models() -> List[str]:
    """
    Return list of available model names.
    """
    return list(models_info.keys())


def get_model_instance(model_name: str):
    """
    Lazy-load and return the model instance for model_name.
    Caches instances in loaded_models.
    """
    if model_name not in models_info:
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models()}")
    if model_name in loaded_models:
        return loaded_models[model_name]

    info = models_info[model_name]
    path = info["path"]
    model_type = info["type"]
    print(f"[model_manager] Loading model '{model_name}' of type '{model_type}' from {path}")
    if model_type == "yolo":
        model = YOLO(str(path))
    elif model_type == "keras":
        model = load_model(str(path))
    else:
        raise RuntimeError(f"Unknown model type '{model_type}' for model '{model_name}'")

    loaded_models[model_name] = model
    return model


def predict_detection(model, image_path: str) -> Dict[str, int]:
    """
    Run YOLO detection model on image_path.
    Assumes class indices: 0=female, 1=male. Returns counts.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read for detection")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    detected = []
    if results and results[0].boxes is not None and hasattr(results[0].boxes, "cls"):
        try:
            cls_arr = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
            detected = cls_arr
        except Exception:
            detected = []
    return {
        "males": detected.count(1),
        "females": detected.count(0)
    }


def preprocess_for_keras(image_path: str, target_size: tuple) -> np.ndarray:
    """
    Read image from image_path, convert BGR->RGB, resize to (width, height) = target_size reversed,
    scale pixel values to [0,1], return batch shape (1, h, w, 3).
    target_size: (h, w)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read for classification")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = target_size
    image_resized = cv2.resize(image_rgb, (w, h))
    img_array = image_resized.astype(np.float32) / 255.0
    batch = np.expand_dims(img_array, axis=0)  # (1, h, w, 3)
    return batch


def predict_classification(model, image_path: str) -> Dict[str, Any]:
    """
    Run Keras classification model on image_path.
    Supports:
      - Output shape (1,1) or (1,) with sigmoid: p_male in [0,1], p_female = 1-p_male
      - Output shape (1,2) with softmax: [p_female, p_male]
    Returns:
      {"prediction": "male"/"female", "probabilities": {"female": float, "male": float}}
    """
    input_shape = model.input_shape  # e.g. (None, h, w, 3)
    if not input_shape or len(input_shape) < 4:
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")
    _, h, w, c = input_shape
    if c != 3:
        raise RuntimeError(f"Expected 3-channel input for classification, got {c} channels")

    batch = preprocess_for_keras(image_path, target_size=(h, w))
    preds = model.predict(batch)
    probs_female = None
    probs_male = None

    if isinstance(preds, np.ndarray):
        arr = preds
    else:
        arr = np.array(preds)

    if arr.ndim == 2 and arr.shape[1] == 1:
        # sigmoid output
        p_male = float(arr[0][0])
        p_female = 1.0 - p_male
        probs_female, probs_male = p_female, p_male
    elif arr.ndim == 1 and arr.shape[0] == 1:
        p_male = float(arr[0])
        p_female = 1.0 - p_male
        probs_female, probs_male = p_female, p_male
    elif arr.ndim == 2 and arr.shape[1] == 2:
        p0 = float(arr[0][0])
        p1 = float(arr[0][1])
        total = p0 + p1
        if total > 0:
            probs_female = p0 / total
            probs_male = p1 / total
        else:
            probs_female, probs_male = 0.5, 0.5
    else:
        raise RuntimeError(f"Unexpected prediction shape from classification model: {arr.shape}")

    prediction = "male" if probs_male >= probs_female else "female"
    return {
        "prediction": prediction,
        "probabilities": {"female": probs_female, "male": probs_male}
    }


def predict(model_name: str, image_path: str) -> Dict[str, Any]:
    """
    Dispatch to detection or classification based on model type.
    Returns:
      {"type": "detection", "result": {"males": int, "females": int}}
    or
      {"type": "classification", "result": {"prediction": str, "probabilities": {...}}}
    """
    if model_name not in models_info:
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models()}")
    model = get_model_instance(model_name)
    mtype = models_info[model_name]["type"]
    if mtype == "yolo":
        counts = predict_detection(model, image_path)
        return {"type": "detection", "result": counts}
    elif mtype == "keras":
        cls_res = predict_classification(model, image_path)
        return {"type": "classification", "result": cls_res}
    else:
        raise RuntimeError(f"Unknown model type '{mtype}' for model '{model_name}'")


# On import, scan available models but do NOT load weights yet.
scan_models()
