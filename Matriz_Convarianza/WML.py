from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
iris = load_iris()
X = np.array(iris.data)
MC = np.dot(X.T,X)
autovalores, W = np.linalg.eig(MC)
idx = autovalores.argsort()[::-1]#Para darle la vulta a un array, argsort() ordena de menor a mayor
autovalores = autovalores[idx]
W = W[:,idx]
Wr = W[:,0:2]
proyeccion = np.dot(X, Wr)
df = pd.DataFrame(proyeccion,columns=['componente1','componente2'])
df['target'] = iris.target
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
plt.scatter(df0['componente1'],df0['componente2'],color ='red',marker='+')
plt.scatter(df1['componente1'],df1['componente2'],color ='green',marker='*')
plt.scatter(df2['componente1'],df2['componente2'],color ='red',marker='.')
plt.xlabel('componente1')
plt.ylabel('componente2')
plt.show()