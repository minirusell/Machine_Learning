from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
print(dir(iris))
df = pd.DataFrame(iris.data, columns= iris.feature_names)
df['target'] = iris.target
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 'columns'),df.target, test_size = 0.2)
resultado = np.zeros(100)
for i in range(100):
    model = RandomForestClassifier(n_estimators=i+1)
    model.fit(X_train,y_train)
    resultado[i] = model.score(X_test, y_test)
print(f"El valor de arboles que da mayor probabilid de acierto es {np.argmax(resultado)} con una de {np.max(resultado)}")
x = np.arange(1, 101)
plt.scatter(x, resultado, color = 'red')
plt.xlabel('numeor de muestra')
plt.ylabel('precicios')
plt.show()
print(resultado)


