
"""
Save this as 'load_onnx_model.py'
Requirements: pip install onnxruntime Pillow numpy
"""

import onnxruntime as ort
import numpy as np
import json
from PIL import Image

class ONNXPotatoDiseaseModel:
    def __init__(self, model_path="/kaggle/working/potato_disease_resnet.onnx"):
        """
        Load ONNX model for potato disease classification.
        
        Args:
            model_path: Path to .onnx model file
        """
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
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Input name: {self.input_name}")
        print(f"✓ Classes: {self.class_names}")
        print(f"✓ Input size: {self.transform_info['input_size']}")
    def preprocess_image(self, image_path):
        """
        Preprocess image for model input with proper float32 conversion.
        """
        image = Image.open(image_path).convert('RGB')
        from PIL import Image as PILImage
        target_size = self.transform_info['input_size']
        image = image.resize((target_size, target_size))
        image_array = np.array(image).astype(np.float32) / 255.0
        mean = np.array(self.transform_info['mean'], dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.transform_info['std'], dtype=np.float32).reshape(1, 1, 3)
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype(np.float32)
        return image_array, image

model = ONNXPotatoDiseaseModel(
    model_path="G:\\python\\python 2025\\thesis\\tazrin\\app\\detection_service\\models\\potato_disease_resnet.onnx"
)

def predict_image_onnx(model, image_path, top_k=3):
    """
    Make prediction on a single image.
    Args:
        model: ONNXPotatoDiseaseModel instance
        image_path: Path to image file
        top_k: Return top k predictions
    Returns:
        Dictionary with predictions
    """
    input_array, original_image = model.preprocess_image(image_path)
    
    print(f"Input shape: {input_array.shape}")
    print(f"Input dtype: {input_array.dtype}")
    print(f"Input range: [{input_array.min():.3f}, {input_array.max():.3f}]")
    
    outputs = model.session.run(None, {model.input_name: input_array})
    logits = outputs[0][0]  
    print(f"Logits shape: {logits.shape}")
    max_logits = np.max(logits)
    exp_logits = np.exp(logits - max_logits)
    probabilities = exp_logits / np.sum(exp_logits)
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    predictions = []
    for idx in top_indices:
        predictions.append({
            'class': model.class_names[idx],
            'confidence': float(probabilities[idx]),
            'percentage': f"{probabilities[idx] * 100:.2f}%"
        })
    
    return {
        'image_path': image_path,
        'top_prediction': predictions[0],
        'all_predictions': predictions,
        'is_healthy': predictions[0]['class'] == 'Potato___healthy' if 'Potato___healthy' in model.class_names else None
    }

def predict_and_display(model, image_path):
    """
    Make prediction and display results.
    
    Args:
        model: ONNXPotatoDiseaseModel instance
        image_path: Path to image file
    """
    import matplotlib.pyplot as plt
    result = predict_image_onnx(model, image_path)
    image = Image.open(image_path)
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {result['top_prediction']['class']}")
    plt.axis('off')
    
    plt.subplot(2, 1, 2)
    classes = [p['class'].replace('Potato___', '') for p in result['all_predictions']]
    confidences = [p['confidence'] * 100 for p in result['all_predictions']]
    
    bars = plt.barh(classes, confidences, color='skyblue')
    plt.xlabel('Confidence (%)')
    plt.title('Top Predictions')
    plt.xlim([0, 100])
    
    
    for bar, conf in zip(bars, confidences):
        plt.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                f'{conf:.1f}%', va='center', ha='right', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return result

def predict_batch(model, image_paths):
    """
    Predict on multiple images at once.
    
    Args:
        model: ONNXPotatoDiseaseModel instance
        image_paths: List of image paths
    
    Returns:
        List of prediction results
    """
    results = []
    for img_path in image_paths:
        try:
            result = predict_image_onnx(model, img_path, top_k=1)
            results.append(result)
            print(f"{img_path}: {result['top_prediction']['class']} ({result['top_prediction']['percentage']})")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results.append({'image_path': img_path, 'error': str(e)})
    
    return results


result = predict_and_display(
    model,
    "/kaggle/input/potato-diseases-datasets/Blackleg/10.jpg"
)

print(result)
