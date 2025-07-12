from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#../dataset/20news-bydate_py3.pkz
data = joblib.load('../dataset/20news-bydate_py3.pkz')
train_data = data['train']
test_data = data['test']

pipeline = make_pipeline(CountVectorizer(),LogisticRegression(max_iter=400))

pipeline.fit(train_data.data, train_data.target)

y_pred = pipeline.predict(test_data.data)

print("Accuracy Score: ", accuracy_score(test_data.target, y_pred))

joblib.dump(pipeline, '../model/lr_pipeline.pkl')