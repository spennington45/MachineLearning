import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap

data = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)

corpus = []
stopwords = stopwords.words("english")
stopwords.remove("not")

for i in range(0, data.index.stop):
    review = re.sub("[^a-zA-Z]", " ", data["Review"][i])
    review = review.lower().split()
    ps = PorterStemmer()
    reviewList = []
    for word in review:
        if not word in set(stopwords):
            reviewList.append(ps.stem(word))
    review = " ".join(reviewList)
    corpus.append(review)

cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
# print(cm)
print("Naive Bayes model")
print(accuracy_score(y_test, y_pred))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
# print(cm)
accScore = accuracy_score(y_test, y_pred)
print("NaRandom Forest model")
print(accScore)
