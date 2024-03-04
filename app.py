import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd


app = Flask(__name__)

# Load the model
model = pickle.load(open('boston_housing_model.sav', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb')) 

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Standardize the data using the scaler made in the DataPreparation.py
    transformed_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = model.predict(transformed_data)

    output = prediction[0]
    return jsonify(output)

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    fianal_input = scaler.transform(np.array(data).reshape(1, -1))
    print(fianal_input)
    prediction = model.predict(fianal_input)[0]
    return render_template('home.html', prediction_text='Predicted House Price: {}'.format(prediction)) 

if __name__ == "__main__":
    app.run(debug=True)