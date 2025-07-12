#../model/lr.pkl
import joblib
import pandas as pd

lr = joblib.load('../model/lr.pkl')


new_data = [[6.4, 2.8, 5.6, 2.2]]
new_pred = lr.predict(new_data)
ground_truth = 2

print(f"New Data: {new_data}")
print(f"New Prediction: {new_pred}")
print(f"Ground Truth: {ground_truth}")