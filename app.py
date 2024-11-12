from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np
import torch  # Or any ML framework you're using
from tensorflow.keras.models import load_model as keras_load_model

app = FastAPI()

# Load the model once at startup
model = keras_load_model("food_recognition_model.h5")
model.eval()  # Set the model to evaluation mode

def preprocess_image(image_path, target_size=(224, 224)):
    # Load and preprocess the image
    image = Image.open(image_path).resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        image = Image.open(io.BytesIO(await file.read()))

        # Preprocess the image (depends on your model requirements)
        preprocessed_image = preprocess_image(image)  # E.g., resizing, normalization, converting to tensor

        # Convert to batch format (if required by the model)
        input_batch = preprocessed_image.unsqueeze(0)  # Add batch dimension if necessary

        # Make the prediction
        with torch.no_grad():  # Disable gradient calculation for inference
            prediction = model(input_batch)

        # Convert prediction to a human-readable format
        predicted_class = torch.argmax(prediction, dim=1).item()  # Assuming classification
        confidence = torch.softmax(prediction, dim=1).max().item()  # Get confidence score

        return {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

