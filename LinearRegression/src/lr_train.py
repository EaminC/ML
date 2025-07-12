from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
#../dataset/diabetes.csv
df = pd.read_csv('../dataset/diabetes.csv')
X  = df.drop('target', axis=1)
y = df['target']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

#test
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

print(f"Mean Squared Error Train: {mean_squared_error(y_train, y_pred_train)}")
print(f"Mean Squared Error Test: {mean_squared_error(y_test, y_pred_test)}")

#save the model
joblib.dump(lr, '../model/lr.pkl')





