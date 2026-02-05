from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
puntuacion_LR = cross_val_score(LogisticRegression(),iris.data,iris.target)
puntuacion_SVC = cross_val_score(SVC(),iris.data,iris.target)
puntuacion_RF = cross_val_score(RandomForestClassifier(),iris.data,iris.target)
print(puntuacion_LR)
print(puntuacion_SVC)
print(puntuacion_RF)