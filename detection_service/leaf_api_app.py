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
import onnxruntime as ort
import json

# Silence TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load leaf model
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

# ONNX Tuber Disease Model Class
class ONNXPotatoDiseaseModel:
    def __init__(self, model_path="./models/potato_disease_resnet.onnx"):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        
        class_names_path = model_path.replace('.onnx', '_classes.json')
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        
        transform_path = model_path.replace('.onnx', '_transform.json')
        with open(transform_path, 'r') as f:
            self.transform_info = json.load(f)
        
        print(f"✓ Tuber Model loaded: {model_path}")
    
    def preprocess_image(self, image: Image.Image):
        target_size = self.transform_info['input_size']
        image = image.resize((target_size, target_size))
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array(self.transform_info['mean'], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.transform_info['std'], dtype=np.float32).reshape(1, 1, 3)
        
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype(np.float32)
        
        return image_array

# Load tuber model
tuber_model = ONNXPotatoDiseaseModel()

# Prediction functions
def predict_leaf_disease(image: Image.Image):
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    return CLASS_NAMES[np.argmax(preds[0])]

def predict_tuber_disease(image: Image.Image, top_k=3):
    input_array = tuber_model.preprocess_image(image)
    outputs = tuber_model.session.run(None, {tuber_model.input_name: input_array})
    logits = outputs[0][0]
    
    max_logits = np.max(logits)
    exp_logits = np.exp(logits - max_logits)
    probabilities = exp_logits / np.sum(exp_logits)
    
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        predictions.append({
            'class': tuber_model.class_names[idx],
            'confidence': float(probabilities[idx]),
            'percentage': f"{probabilities[idx] * 100:.2f}%"
        })
    
    return {
        'top_prediction': predictions[0],
        'all_predictions': predictions,
        'is_healthy': predictions[0]['class'] == 'Potato___healthy' if 'Potato___healthy' in tuber_model.class_names else None
    }

# FastAPI app
app = FastAPI(title="Potato Disease Detection API")

@app.get("/health")
async def health_check():
    return {"status": "ok by abir"}

# Leaf Disease Detection Endpoints
@app.post("/predict_leaf")
async def predict_leaf(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        disease_name = predict_leaf_disease(image)
        return {"disease": disease_name}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_leaf_base64")
async def predict_leaf_base64(data: dict = Body(...)):
    try:
        print(f"Received data: {data}")
        img_base64 = data.get("image")
        print(f"img_base64 length: {len(img_base64) if img_base64 else 0}")
        if not img_base64:
            return {"error": "No image provided"}
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        disease_name = predict_leaf_disease(image)
        return {"disease": disease_name}
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Tuber Disease Detection Endpoints
@app.post("/predict_tuber")
async def predict_tuber(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = predict_tuber_disease(image)
        return result
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_tuber_base64")
async def predict_tuber_base64(data: dict = Body(...)):
    try:
        print(f"Received data: {data}")
        img_base64 = data.get("image")
        print(f"img_base64 length: {len(img_base64) if img_base64 else 0}")
        if not img_base64:
            return {"error": "No image provided"}
        img_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = predict_tuber_disease(image)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# Run locally on port 8001
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
