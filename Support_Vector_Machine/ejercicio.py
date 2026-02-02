from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

numeros = load_digits()
#print(dir(numeros))
#print(numeros.feature_names)
Datos = np.array(numeros.data)
df = pd.DataFrame(numeros.data, columns=numeros.feature_names)
df['target'] = numeros.target

X = df.drop(['target'], axis = 'columns')
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
model = SVC()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))





