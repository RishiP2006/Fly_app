import os
import tempfile
import traceback
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.model.model_manager import available_models, predict as run_predict

# FastAPI app instance
app = FastAPI(
    title="Drosophila Gender Detection API",
    description="Upload an image and select a model to detect or classify drosophila gender",
    version="1.0.0"
)

# Enable CORS (allowing frontend like Streamlit to access this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running ✅"}

@app.get("/models")
def get_models():
    """
    Return a list of available model names.
    Example response: {"models": ["best", "drosophila_classifier"]}
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
      {
        "type": "detection" or "classification",
        "result": {...}
      }
    """
    # Check if model is available
    try:
        models = available_models()
        if model_name not in models:
            return JSONResponse(
                content={"error": f"Model '{model_name}' not found. Available: {models}"},
                status_code=400
            )
    except Exception as e:
        return JSONResponse(content={"error": f"Could not list models: {e}"}, status_code=500)

    tmp_path = None
    try:
        # Save the uploaded image temporarily
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Run the prediction
        result = run_predict(model_name, tmp_path)
        return JSONResponse(content=result)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict_endpoint] Exception: {tb}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        # Delete temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
