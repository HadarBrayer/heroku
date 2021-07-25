import requests
import pandas as pd
import numpy as np
import json

X_test = pd.read_csv('X_test.csv')
preds = np.loadtxt('preds.csv')

my_dict = {feature: 0 for feature in X_test.columns}
my_list = [feature for feature in X_test.columns]

for i in range(5):
    for j in range(len(X_test.columns)):
        my_dict[my_list[j]] = X_test.iloc[i][my_list[j]]
    r = requests.get('https://heroku12hb.herokuapp.com/predict_churn', params=my_dict)
    print(r.url, preds[i])

# bonus question
json_params = X_test[0:5].to_json(orient='records')  # for post in the inference we need json
response = requests.post('https://heroku12hb.herokuapp.com/predict_churn_bulk', json=json_params).json()
predictions = np.array([sample['pred'] for sample in response])  # for pred in each dict in the json
print('Predictions by the loaded model: ', predictions)
print('Predictions from the csv file: ', preds[0:5])
assert np.array_equal(preds[0:5], predictions)
print('Predictions are the same by the model and the csv file')
