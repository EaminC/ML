#../model/lr.pkl
import joblib
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
target_names = fetch_20newsgroups(subset='train').target_names
lr_pipeline = joblib.load('../model/lr_pipeline.pkl')


new_data = ['NASA s latest spacecraft successfully launched yesterday.',
            'My computer crashes every time I boot Linux.']
new_pred = lr_pipeline.predict(new_data)
#all name

new_pred_names = [target_names[i] for i in new_pred]
print(f"New Data: {new_data}")
print(f"New Prediction: {new_pred}")
print(f"New Prediction Name: {new_pred_names}")
