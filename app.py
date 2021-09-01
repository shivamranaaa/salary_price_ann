import numpy as np
import tensorflow 
from tensorflow import keras

from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = keras.models.load_model("yo.h5", custom_objects=None, compile=True, options=None)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [(int_features)]
    prediction = model.predict(final_features)

    #output = round(prediction[0][0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(round(prediction[0][0], 2)))


if __name__ == "__main__":
    app.run()
