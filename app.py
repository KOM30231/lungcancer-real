from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("model.h5")

# Template for the upload form
UPLOAD_FORM = '''
<!doctype html>
<title>Lung Cancer Predictor</title>
<h2>Upload an image (JPG/PNG):</h2>
<form action="/predict" method="get" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*">
  <input type="submit" value="Upload and Predict">
</form>
{% if result %}
  <h3>Prediction: {{ result }}</h3>
  <h4>Label: {{ label }}</h4>
  <h4>Confidence: {{ confidence }}</h4>
{% endif %}
'''

@app.route('/', methods=['GET'])
def home():
    return render_template_string(UPLOAD_FORM)

@app.route('/predict', methods=['GET'])
def predict_get():
    if 'image' not in request.files:
        return render_template_string(UPLOAD_FORM + "<p style='color:red;'>No image uploaded</p>")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template_string(UPLOAD_FORM + "<p style='color:red;'>Empty filename</p>")

    try:
        # Save uploaded file
        filename = "uploaded_image.jpg"
        image_file.save(filename)

        # Preprocess image
        img = Image.open(filename).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Map result to label
        labels = ["Normal", "Benign", "Malignant"]
        label_name = labels[result] if result < len(labels) else "Unknown"

        return render_template_string(UPLOAD_FORM,
                                      result=result,
                                      label=label_name,
                                      confidence=f"{confidence:.2f}")

    except Exception as e:
        return render_template_string(UPLOAD_FORM + f"<p style='color:red;'>Error: {str(e)}</p>")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
