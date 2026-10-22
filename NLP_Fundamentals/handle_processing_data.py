from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

documents = [
    "I love programming in Python",
    "Java is a powerful language",
    "Python is great for data science",
    "I prefer Java for system programming",
    "Data science is fun with Python"
]

# Labels (0: Python-related, 1: Java-related)
labels = [0, 1, 0, 1, 0]


vectorizer = CountVectorizer()

X = vectorizer.fit(documents)
print(X.vocabulary_)
X = vectorizer.transform(documents)
print(X)
print(X.toarray())
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

classifier = MultinomialNB()

classifier.fit(x_train, y_train)

pred = classifier.predict(x_test)

print(accuracy_score(y_test, pred))