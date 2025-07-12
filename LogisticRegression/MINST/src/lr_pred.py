from sklearn.datasets import fetch_openml
import numpy as np
import joblib
import matplotlib.pyplot as plt

MINST_binary = fetch_openml('mnist_784',data_home= '../dataset/')
img_index = 4
img = np.array(MINST_binary.data)[img_index]
print(img.shape)
lr = joblib.load('../model/lr_mnist.pkl')
y_pred = lr.predict([img])


print("Ground Truth: ", MINST_binary.target[img_index])
print("Prediction: ", y_pred)



plt.imshow(img.reshape(28,28), cmap='gray')
plt.show()
