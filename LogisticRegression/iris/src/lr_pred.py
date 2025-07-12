#../model/lr.pkl
import joblib
import pandas as pd

lr_ovr = joblib.load('../model/lr_ovr.pkl')
lr_multinomial = joblib.load('../model/lr_multinomial.pkl')


new_data = [[6.4, 2.8, 5.6, 2.2]]
new_pred_multinomial = lr_multinomial.predict(new_data)
new_pred_ovr = lr_ovr.predict(new_data)
ground_truth = 2

print(f"New Data: {new_data}")
print(f"New Prediction Multinomial: {new_pred_multinomial}")
print(f"New Prediction OVR: {new_pred_ovr}")
print(f"Ground Truth: {ground_truth}")