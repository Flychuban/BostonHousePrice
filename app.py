import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd


app = Flask(__name__)

# Load the model
model = pickle.load(open('boston_housing_model.sav', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)