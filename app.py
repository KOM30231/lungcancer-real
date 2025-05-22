from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model("lung_cancer_model.h5")

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Load and preprocess image
        file = request.files['image']
        img = Image.open(file).convert("RGB").resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

        # Predict
        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))  # assuming softmax output

        # Optional: Map prediction to class label
        labels = ["Normal", "Benign", "Malignant"]
        label_name = labels[result] if result < len(labels) else "Unknown"

        return jsonify({
            'prediction': result,
            'label': label_name,
            'confidence': float(np.max(prediction))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Home route
@app.route('/')
def home():
    return "✅ Lung Cancer Predictor API is running!"

# Run locally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
