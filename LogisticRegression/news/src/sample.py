from sklearn.feature_extraction.text import CountVectorizer

sample_data = ['The quick brown fox jumps over the lazy dog', 
               'This is a sample sentence', 
               'This is another sentence']

CV = CountVectorizer()

X = CV.fit_transform(sample_data)

print(X.toarray())

print(CV.get_feature_names_out())