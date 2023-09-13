from flask import Flask, request, jsonify
import os
import pickle
from sklearn.model_selection import cross_val_score
import pandas as pd


os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config['DEBUG'] = True

@app.route("/", methods=['GET'])
def hello():
    return "Bienvenido a mi API del modelo"

@app.route('/predict', methods=['POST'])
def predict():
    model = pickle.load(open('data/model_acoruña','rb'))

    req = request.get_json()
    predict_data = []
        # chequear location
    province_name = req.pop('province')
    for val in req.values():
        predict_data.append(int(val))
    
    model = pickle.load('rf.regressor.pkl')
    prediction = model.predict(predict_data)

    return prediction[0]

app.run()