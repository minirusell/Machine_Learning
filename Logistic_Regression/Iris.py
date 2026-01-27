from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

iris = load_iris()
#print(dir(iris))
#print(iris.data[0])
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,train_size=0.2)
#print(help(train_test_split))
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
y_predicted = model.predict(X_test)
print(confusion_matrix(y_test,y_predicted))


