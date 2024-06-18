from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__)

# Path to the H5 file
model_path = os.path.join(os.path.dirname(__file__), 'keras_model.h5')

# Load the Keras model
model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
    image_array = np.asarray(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    image = Image.open(file.stream)
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    result = 'Autism Detected' if predictions[0][0] > 0.5 else 'Non-Autism'
    return result

if __name__ == '__main__':
    app.run(debug=True)
