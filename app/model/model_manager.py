# app/model/model_manager.py

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
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            if suffix == ".pt":
                model_type = "yolo"
            elif suffix in (".keras", ".h5"):
                model_type = "keras"
            else:
                # skip unsupported file types
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
    - For YOLO: returns a ultralytics.YOLO instance.
    - For Keras: returns a tf.keras.Model loaded from disk.
    Caches the instance in loaded_models.
    """
    if model_name not in models_info:
        raise ValueError(f"Model '{model_name}' not found. Available: {available_models()}")
    # If already loaded, return it
    if model_name in loaded_models:
        return loaded_models[model_name]
    info = models_info[model_name]
    path = info["path"]
    model_type = info["type"]
    print(f"[model_manager] Loading model '{model_name}' of type '{model_type}' from {path}")
    if model_type == "yolo":
        # Load YOLO detection model
        model = YOLO(str(path))
    elif model_type == "keras":
        # Load Keras model
        # Configure TF to allow growth if needed, etc.
        # Example: limit GPU memory growth if GPUs exist. On CPU-only, it's fine.
        # tf.config.run_functions_eagerly(True)  # optional
        model = load_model(str(path))
    else:
        raise RuntimeError(f"Unknown model type '{model_type}' for model '{model_name}'")
    loaded_models[model_name] = model
    return model


def predict_detection(model, image_path: str) -> Dict[str, int]:
    """
    Run YOLO detection model on image at image_path.
    Assumes class indices: 0=female, 1=male. Returns counts.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read for detection")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    detected = []
    # Check results
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
    Read image from image_path, convert BGR->RGB, resize to target_size (width, height),
    scale pixel values to [0,1], and return a batch of shape (1, h, w, c).
    - target_size: (height, width) or (None, None) if uncertain.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be read for classification")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize
    h, w = target_size
    # cv2.resize expects (width, height) in size tuple
    image_resized = cv2.resize(image_rgb, (w, h))
    # Scale to [0,1]
    img_array = image_resized.astype(np.float32) / 255.0
    # Expand dims to batch
    batch = np.expand_dims(img_array, axis=0)  # shape (1, h, w, 3)
    return batch


def predict_classification(model, image_path: str) -> Dict[str, Any]:
    """
    Run Keras classification model on image_path.
    Assumes model outputs either:
      - shape (1, 1) with sigmoid activation → binary output in [0,1], where ≥0.5 => class 1 (male), <0.5 => class 0 (female).
      - shape (1, 2) with softmax → two probabilities [p0, p1], where index 0=female, 1=male.
    Returns a dict with keys:
      "prediction": "male" or "female",
      "probabilities": {"female": <float>, "male": <float>}
    """
    # Determine model input shape: model.input_shape is typically (None, h, w, c)
    input_shape = model.input_shape  # tuple, e.g. (None, 224, 224, 3)
    if not input_shape or len(input_shape) < 4:
        raise RuntimeError(f"Unexpected model input shape: {input_shape}")
    _, h, w, c = input_shape
    if c != 3:
        # If grayscale or other channel number, adapt as needed; for now, we expect 3 channels.
        raise RuntimeError(f"Expected 3-channel input for classification, got {c} channels")
    # Preprocess image
    batch = preprocess_for_keras(image_path, target_size=(h, w))  # returns shape (1, h, w, 3)
    # Run prediction
    preds = model.predict(batch)
    # preds shape could be (1,1) or (1,2) or (1,) etc.
    # Normalize into female/male probabilities
    # Case 1: shape (1,1) or (1,): sigmoid
    # Case 2: shape (1,2): softmax
    probs_female = None
    probs_male = None
    if preds.ndim == 2 and preds.shape[1] == 1:
        # sigmoid: preds in [[p]] where p = probability of class 1 (male). female prob = 1-p
        p_male = float(preds[0][0])
        p_female = 1.0 - p_male
        probs_female = p_female
        probs_male = p_male
    elif preds.ndim == 1 and preds.shape[0] == 1:
        # maybe shape (1,), treat same as (1,1)
        p_male = float(preds[0])
        p_female = 1.0 - p_male
        probs_female = p_female
        probs_male = p_male
    elif preds.ndim == 2 and preds.shape[1] == 2:
        # softmax or logits; assume model includes softmax, so values sum ~1
        p0 = float(preds[0][0])
        p1 = float(preds[0][1])
        # if not exactly sum 1 (logits), you may apply softmax; here we assume it’s probabilities.
        total = p0 + p1
        if total > 0:
            probs_female = p0 / total
            probs_male = p1 / total
        else:
            # fallback: equal
            probs_female = 0.5
            probs_male = 0.5
    else:
        # Unexpected output shape
        raise RuntimeError(f"Unexpected prediction shape from classification model: {preds.shape}")
    # Determine predicted class
    prediction = "male" if probs_male >= probs_female else "female"
    return {
        "prediction": prediction,
        "probabilities": {
            "female": probs_female,
            "male": probs_male
        }
    }


def predict(model_name: str, image_path: str) -> Dict[str, Any]:
    """
    Dispatch prediction based on model type.
    Returns a dict of the form:
      {
        "type": "detection",
        "result": {"males": int, "females": int}
      }
    or
      {
        "type": "classification",
        "result": { "prediction": "...", "probabilities": {...} }
      }
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
