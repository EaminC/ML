from sklearn import datasets
import pandas as pd
#load the dataset
data = datasets.load_iris()
#save the data to a csv file

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.to_csv('iris.csv', index=False)