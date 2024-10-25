from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load the H5 model
model = tf.keras.models.load_model('/image_classification_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = data.get('input')
    prediction = model.predict([input_data])
    prediction = prediction.tolist()  # Convert to list for JSON response

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(port=5000)