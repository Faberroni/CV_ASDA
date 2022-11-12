import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
from keras.utils import img_to_array

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications


MODEL_PATH = 'models/model_cnn.h5'

model = load_model(MODEL_PATH)

print('Model loaded. Start serving...')


def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        # Process your result for human
        # Percentange result
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        # Labeled result
        classes = ['miner', 'rust']
        MaxPosition=np.argmax(preds)  
        prediction_label=classes[MaxPosition]

        # Serialize the result, you can add additional fields
        return jsonify(result=prediction_label, probability=pred_proba)

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)


#tes buat git