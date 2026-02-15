import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import onnxruntime as ort

# Model paths (relative to the app directory)
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'detection_service', 'models')
LEAF_MODEL_PATH = os.path.join(MODEL_DIR, 'efficientnet_potato_leaf_model_final.keras')
TUBER_MODEL_PATH = os.path.join(MODEL_DIR, 'resnet_tuber_disease_model.onnx')

# Global model variables
leaf_model = None
tuber_model = None

# Leaf disease class names
LEAF_CLASS_NAMES = [
    "Bacteria",
    "Fungi",
    "Healthy",
    "Nematode",
    "Pest",
    "Phytopthora",
    "Virus"
]
LEAF_IMAGE_SIZE = 256

class ONNXPotatoDiseaseModel:
    def __init__(self, model_path):
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

        print(f"Tuber model loaded: {model_path}")

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

def load_models():
    global leaf_model, tuber_model
    if leaf_model is None:
        leaf_model = load_model(LEAF_MODEL_PATH, compile=False)
        print("Leaf model loaded")
    if tuber_model is None:
        tuber_model = ONNXPotatoDiseaseModel(TUBER_MODEL_PATH)
        print("Tuber model loaded")

def predict_leaf_disease(image: Image.Image):
    load_models()
    img = image.resize((LEAF_IMAGE_SIZE, LEAF_IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = leaf_model.predict(img_array, verbose=0)
    return LEAF_CLASS_NAMES[np.argmax(preds[0])]

def predict_tuber_disease(image: Image.Image, top_k=3):
    load_models()
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
