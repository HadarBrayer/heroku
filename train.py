import pandas as pd
import pickle
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv("cellular_churn_greece.csv")
features = list(df.columns)
target = 'churned'
features.remove(target)
X = df[features]
y = df[target]


# Split the data into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

# Choose a model type of your choice.
clf = RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# print how well the model is predicting on the test set.
clf_report = classification_report(y_test, y_pred)
print(clf_report)

# Using Pickle, save your model to a file on disk named churn_model.pkl
filename = 'churn_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

# Your X_test as a csv file X_test.csv (hint: don’t need to save the index - use
# index=False in Pandas DataFrame to_csv method)
X_test.to_csv('X_test.csv',index=False)

# The predictions of your model on X_test as a csv file preds.csv (hint: check out Numpy savetxt method)
np.savetxt('preds.csv', y_pred, delimiter=',')

