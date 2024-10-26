from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('image_classification_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is in request
    if 'input' not in request.files:
        return jsonify({"error": "File not found in request"}), 400

    # Get the image file from request
    file = request.files['input']

    try:
        image = Image.open(file).resize((150, 150))  # Resize for model input
        image = np.array(image) / 255.0  # Normalize if needed
        image = np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 400

    try:
        prediction = model.predict(image)  # Make prediction
        prediction = prediction.tolist()
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    return jsonify({'prediction': prediction})

if __name__ == '__main__':  # Fixed __name_
    app.run(port=5000)