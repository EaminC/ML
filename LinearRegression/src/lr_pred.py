#../model/lr.pkl
import joblib
import pandas as pd

lr = joblib.load('../model/lr.pkl')

#predict new data
new_data = [[0.03807591, 0.05068012, 0.06169621, -0.01882338, 0.04327843, -0.00259226, 0.01990842, -0.01764613, -0.05225110, -0.04416530]]
new_pred = lr.predict(new_data)
ground_truth = 151.0

print(f"New Data: {new_data}")
print(f"New Prediction: {new_pred}")
print(f"Ground Truth: {ground_truth}")