import pickle
import requests
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
import json
import flask
import os
from flask import jsonify, make_response



app = Flask(__name__)

filename = 'churn_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

# http://127.0.0.1:5000/predict_churn?is_male=0&num_inters=0&late_on_payment=1&age=33&years_in_contract=4.36
@app.route('/predict_churn')
def predict_churn():
    if globals()['loaded_model']:
        pass
    else:
        loaded_model = pickle.load(open(filename, 'rb'))
    is_male = request.args.get("is_male")
    num_inters = request.args.get("num_inters")
    late_on_payment =request.args.get("late_on_payment")
    age = request.args.get("age")
    years_in_contract = request.args.get("years_in_contract")
    data = np.array([is_male, num_inters, late_on_payment, age, years_in_contract])
    print(data)
    pred = loaded_model.predict(data.reshape(1,-1))
    return str(pred[0])

# http://127.0.0.1:5000/predict_churn_bulk
@app.route('/predict_churn_bulk', methods=['POST'])
def predict_churn_bulk():
    if globals()['loaded_model']:
        pass
    else:
        loaded_model = pickle.load(open(filename, 'rb'))
    bulk_data = json.loads(flask.request.get_json())
    X_test_bulk = pd.DataFrame(bulk_data)
    prediction = loaded_model.predict(X_test_bulk)
    X_test_bulk['prediction'] = prediction
    return flask.jsonify(X_test_bulk.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT')))
    #app.run(host='127.0.0.1', port=5000)
