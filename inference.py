import pickle
import pandas as pd
import numpy as np
from flask import Flask, request
import time
import os

start_time = time.time()
# Read the model with Pickle from churn_model.pkl
filename = 'churn_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))


app = Flask(__name__)
@app.route('/')
def say_hi():
    return "hi"

#http://127.0.0.1:5000/predict_churn?is_male=0&num_inters=0&late_on_payment=1&age=33&years_in_contract=4.36
# @app.route('/predict_churn')
# def predict_churn():
#     is_male = request.args.get("is_male")
#     num_inters = request.args.get("num_inters")
#     late_on_payment = request.args.get("late_on_payment")
#     age = request.args.get("age")
#     years_in_contract = request.args.get("years_in_contract")
#     data = np.array([is_male, num_inters, late_on_payment, age, years_in_contract])
#     pred = loaded_model.predict(data.reshape(1, 5))
#     return str(pred[0])



if __name__ == '__main__':
    # Heroku provides environment variable 'PORT' that should be listened on by Flask
    port = os.environ.get('PORT')

    if port:
        # 'PORT' variable exists - running on Heroku, listen on external IP and on given by Heroku port
        app.run(host='0.0.0.0', port=int(port))
    else:
        # 'PORT' variable doesn't exist, running not on Heroku, presumabely running locally, run with default
        #   values for Flask (listening only on localhost on default Flask port)
        app.run()



