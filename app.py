import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np

app = FastAPI()

try:
    import tensorflow as tf

    model_path = "food_recognition_model"
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found at {model_path}")
    model = tf.keras.models.load_model(model_path)
except ImportError:
    print("TensorFlow import failed. Running in limited mode.")
    model = None
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

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
    return {"message": "Food Recognition API is running", "model_loaded": model is not None}