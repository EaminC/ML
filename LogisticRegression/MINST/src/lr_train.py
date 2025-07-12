from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import joblib

from sklearn.datasets import fetch_openml
MINST_binary = fetch_openml('mnist_784',data_home= '../dataset/')



scaler = StandardScaler()
X = scaler.fit_transform(MINST_binary.data)

X_train, X_test, y_train, y_test = train_test_split(X, MINST_binary.target, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy Score: ", accuracy_score(y_test, y_pred))

joblib.dump(lr, '../model/lr_mnist.pkl')


