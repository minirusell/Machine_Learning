from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

digitos = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digitos.data,digitos.target,test_size=0.3)

lr = LogisticRegression()
lr.fit(X_train,y_train)
lr.score(X_test, y_test)
svm = SVC()
svm.fit(X_train,y_train)
svm.score(X_test, y_test)
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train,y_train)
rf.score(X_test, y_test)
kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)


scores_l = []
scores_smv = []
scores_rf = []
for train_index, test_index in kf.split(digitos.data):
    X_train, X_test, y_train, y_test = digitos.data[train_index], digitos.data[test_index],\
                                       digitos.target[train_index],digitos.target[test_index]
    scores_l.append(get_score(LogisticRegression(),X_train, X_test, y_train, y_test))
    scores_smv.append(get_score(SVC(),X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(),X_train, X_test, y_train, y_test))
print(scores_l)
print(scores_smv)
print(scores_rf)

print(cross_val_score(LogisticRegression(), digitos.data, digitos.target))
