# app/main.py

import os
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.model.model_manager import available_models, predict as run_predict

app = FastAPI(
    title="Drosophila Gender Detection API",
    description="Upload an image and select a model to detect or classify drosophila gender",
    version="1.0.0"
)

# CORS: in development allow all; in production, restrict to your frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/models")
def get_models():
    """
    Return list of available model names.
    Response: {"models": [<str>, ...]}
    """
    try:
        models = available_models()
        return JSONResponse(content={"models": models})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """
    Accepts:
      - file: image file upload
      - model_name: name from /models
    Returns JSON:
      {"type": "...", "result": {...}}
    or error.
    """
    # Validate model_name
    try:
        models = available_models()
    except Exception as e:
        return JSONResponse(content={"error": f"Could not list models: {e}"}, status_code=500)

    if model_name not in models:
        return JSONResponse(
            content={"error": f"Model '{model_name}' not found. Available: {models}"},
            status_code=400
        )

    tmp_path = None
    try:
        # Save uploaded file to a temporary file
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run prediction
        res = run_predict(model_name, tmp_path)
        # res is a dict: {"type": ..., "result": {...}}
        return JSONResponse(content=res)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict_endpoint] Exception: {tb}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Clean up
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
