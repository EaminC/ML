from sklearn.datasets import fetch_20newsgroups
import pandas as pd


train_data = fetch_20newsgroups(subset='train',data_home= '../dataset/')
test_data = fetch_20newsgroups(subset='test',data_home= '../dataset/')

