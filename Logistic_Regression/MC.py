import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
digits = load_digits()
plt.gray()
plt.matshow(digits.images[0])
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size = 0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.predict([digits.data[67]])
y_predicted = model.predict(X_test)
print(confusion_matrix(y_test,y_predicted))


