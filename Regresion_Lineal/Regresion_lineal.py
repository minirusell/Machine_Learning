import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('canada_per_capita_income.csv')
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['year']],df.per_capita_income)
print(reg.predict([[2024]]))
plt.scatter(df.year,df.per_capita_income, color = 'red', marker = '+')
plt.plot(df.year,reg.predict(df[['year']]))
plt.xlabel('Año')
plt.ylabel('Ingreso per capita')
plt.show()
print(f"la pendiente es{reg.coef_} y el intercepto{reg.intercept_}")

#####################################
#Ajuste lineal por minimos cuadrados#
#####################################
X = np.ones((len(df.year),2))
X[:,0] = df.year
XX = np.dot(X.T,X)
invXX = np.linalg.inv(XX)
beta = np.dot(invXX , np.dot(X.T, df.per_capita_income))
print(f"los valores de la recta que mejor se ajustan son un pendiente de {beta[0]} y un intercepto de {beta[1]}")



"""
df = pd.read_excel("Hoja_pandas.xlsx")
print(df)

reg = linear_model.LinearRegression()
reg.fit(df[['area']],df.precio)#Fit se encarga de entrenar al modelo, obteniendo los parametros que minimizan el error
#Tiene como entrada el conjutno de datos que debe ser una matriz, un vector con los datos clasificados
print(reg.predict([[3300]]))#Predict se encarge de predecir con datos desconodicos
m=reg.coef_
b=reg.intercept_
x=np.linspace
d = pd.read_csv("Prediccion_valores.csv")
p = reg.predict(d)
d['prices'] = p
print(d)
d.to_csv("Prediccion_valores")

plt.scatter(df.area,df.precio, color = 'red',marker='+')
plt.plot(df.area,reg.predict(df[['area']]),color='blue')
plt.xlabel('area')
plt.ylabel('precio')
plt.show()
print(reg.predict(df[['area']]))

"""





