"""
FastAPI REST API for Bangladeshi Taka Note Detection
Serves the trained YOLOv8 model via a /predict endpoint.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from ultralytics import YOLO

# --- Load model once at startup ---
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at '{MODEL_PATH}'")

model = YOLO(MODEL_PATH)

# --- Create FastAPI app ---
app = FastAPI(
    title="Bangladeshi Taka Note Detection API",
    description="Upload an image of Bangladeshi currency and get detected denominations, confidence scores, and bounding boxes.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Bangladeshi Taka Detection API is running", "status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Detect Bangladeshi Taka notes in an uploaded image.

    - **file**: Image file (JPEG or PNG)

    Returns detected denomination names, confidence scores,
    and bounding box coordinates.
    """

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only JPEG and PNG are accepted.",
        )

    # Read the uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    # Run inference
    results = model.predict(source=image, conf=0.25, verbose=False)[0]

    # Build response
    detections = []
    for box in results.boxes:
        detections.append({
            "class": results.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [int(x) for x in box.xyxy[0].tolist()],
        })

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
