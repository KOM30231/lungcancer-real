@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        print("No image found in request")
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        file = request.files['image']
        print("Image received")

        img = Image.open(file).convert("RGB").resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        result = int(np.argmax(prediction))

        print("Prediction complete:", result)
        return jsonify({'prediction': result})
    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({'error': str(e)}), 500
