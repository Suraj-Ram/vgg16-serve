from flask import Flask, render_template, request
from keras.applications.vgg16 import VGG16
from keras.applications import vgg16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import keras
import os

app = Flask(__name__)
app.config["DEBUG"] = True

#model = VGG16()


def loadImage(file):
    image = load_img(file, target_size=(224, 224))
    image = img_to_array(image)
    image = vgg16.preprocess_input(image)

    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    return image


def runModel(file):
    model = VGG16()
    preds = model.predict(loadImage(file))
    labels = vgg16.decode_predictions(preds)
    return labels


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        file = request.files['file']
        print(file.filename)
        file.save(file.filename)

        keras.backend.clear_session()
        result = runModel(file.filename)

        os.remove(file.filename)

        return str(result)

@app.route('/api')
def api():
        return 'works'


app.run(host='0.0.0.0')
