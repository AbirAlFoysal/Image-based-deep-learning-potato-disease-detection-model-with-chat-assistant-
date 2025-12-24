# --- FastAPI (Localhost only, port 8001) ---

from fastapi import FastAPI, File, UploadFile, Body
from PIL import Image
import numpy as np
import io
import os
import base64
import tensorflow as tf
from keras.models import load_model
import uvicorn

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load model
MODEL_PATH = "./models/efficientnet_potato_leaf_model.keras"
model = load_model(MODEL_PATH)

CLASS_NAMES = [
    "Bacteria",
    "Fungi",
    "Healthy",
    "Nematode",
    "Pest",
    "Phytopthora",
    "Virus"
]
IMAGE_SIZE = 256

# Prediction function
def predict_disease(image: Image.Image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    return CLASS_NAMES[np.argmax(preds[0])]

# FastAPI app
app = FastAPI(title="Potato Leaf Disease Classifier")

@app.get("/health")
async def health_check():
    return {"status": "ok by abir"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        disease_name = predict_disease(image)
        return {"disease": disease_name}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_base64")
async def predict_base64(data: dict = Body(...)):
    try:
        print(f"Received data: {data}")
        img_base64 = data.get("image")
        print(f"img_base64 length: {len(img_base64) if img_base64 else 0}")
        if not img_base64:
            return {"error": "No image provided"}
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        disease_name = predict_disease(image)
        return {"disease": disease_name}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Run locally on port 8001
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
