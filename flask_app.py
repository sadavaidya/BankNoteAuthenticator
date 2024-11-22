import pickle
from flask import Flask, request
import pandas as pd
import numpy as np

pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

app = Flask(__name__)

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict')
def predict_note_authentication():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')

    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])

    return ("Prediction is" + str(prediction))






if __name__ == '__main__':
    app.run()
