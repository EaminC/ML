from sklearn.datasets import fetch_20newsgroups

#load the data
train_data = fetch_20newsgroups(subset='train')
test_data = fetch_20newsgroups(subset='test')

