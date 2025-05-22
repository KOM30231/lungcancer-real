from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("lung_cancer_model.h5")

@app.route('/')
def home():
    return "✅ Lung Cancer Predictor API is running!"

# GET-based prediction using local file
@app.route('/predict', methods=['GET'])
def predict_get():
    filename = request.args.get('file')

    if not filename:
        return jsonify({'error': 'No file parameter provided'}), 400

    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(file_path):
        return jsonify({'error': f'File not found: {filename}'}), 404

    try:
        img = Image.open(file_path).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        labels = ["Normal", "Benign", "Malignant"]
        label_name = labels[result] if result < len(labels) else "Unknown"

        return jsonify({
            'prediction': result,
            'label': label_name,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
