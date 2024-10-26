from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('image_classification_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah ada file di request
    if 'input' not in request.files:
        return jsonify({"error": "File tidak ditemukan di permintaan"}), 400

    # Ambil file gambar dari permintaan
    file = request.files['input']
    image = Image.open(file).resize((224, 224))  # Sesuaikan ukuran dengan input model
    image = np.array(image) / 255.0  # Normalisasi jika perlu
    image = np.expand_dims(image, axis=0)  # Tambahkan batch dimension

    prediction = model.predict(image)  # Prediksi
    prediction = prediction.tolist()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000)