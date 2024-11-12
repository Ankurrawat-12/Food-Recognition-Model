import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the model once at startup
model_path = "food_recognition_model"
if not os.path.exists(model_path):
    raise RuntimeError(f"Model not found at {model_path}")

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1).item()
        confidence = np.max(prediction).item()

        return JSONResponse({
            "predicted_class": int(predicted_class),
            "confidence": float(confidence)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Food Recognition API is running"}