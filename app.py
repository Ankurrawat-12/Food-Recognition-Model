import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI()

# Initialize model as None
model = None

try:
    # Load the preprocessing function
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input


    # Custom load model function with the same architecture
    def create_model():
        pretrained_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        pretrained_model.trainable = False

        inputs = pretrained_model.input
        x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        outputs = tf.keras.layers.Dense(101, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)


    # Load the model weights
    model_path = "food_recognition_model.h5"
    if os.path.exists(model_path):
        # Create model with the same architecture
        model = create_model()
        # Load weights
        model.load_weights(model_path)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model loaded successfully")
    else:
        print(f"Model not found at {model_path}")
except ImportError:
    print("TensorFlow import failed. Running in limited mode.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    import traceback

    traceback.print_exc()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not available")

    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.resize((224, 224))
        image_array = np.array(image)

        # Apply MobileNetV2 preprocessing
        image_array = preprocess_input(image_array)
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
    if model is None:
        return {
            "message": "Food Recognition API is running, but model failed to load",
            "model_loaded": False,
            "error": "Check server logs for details"
        }
    return {"message": "Food Recognition API is running", "model_loaded": True}