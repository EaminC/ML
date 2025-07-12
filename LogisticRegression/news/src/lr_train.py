from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import joblib
#../dataset/diabetes.csv
df = pd.read_csv('../dataset/iris.csv')
X  = df.drop('target', axis=1)
y = df['target']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#train the model
lr_ovr = LogisticRegression(multi_class='ovr')#other parameters: penalty='l2', C=1.0, solver='liblinear' max_iter=100
lr_ovr.fit(X_train, y_train)

#test
y_pred_train = lr_ovr.predict(X_train)
y_pred_test = lr_ovr.predict(X_test)

print(f"Accuracy Score Train OVR: {accuracy_score(y_train, y_pred_train)}")
print(f"Accuracy Score Test OVR: {accuracy_score(y_test, y_pred_test)}")
#save the model
joblib.dump(lr_ovr, '../model/lr_ovr.pkl')


#train the model
lr_multinomial = LogisticRegression(multi_class='multinomial')
lr_multinomial.fit(X_train, y_train)

#test
y_pred_train = lr_multinomial.predict(X_train)
y_pred_test = lr_multinomial.predict(X_test)

print(f"Accuracy Score Train Multinomial: {accuracy_score(y_train, y_pred_train)}")
print(f"Accuracy Score Test Multinomial: {accuracy_score(y_test, y_pred_test)}")
#save the model
joblib.dump(lr_multinomial, '../model/lr_multinomial.pkl')












