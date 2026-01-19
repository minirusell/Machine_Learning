import pandas as pd
import numpy as np
from sklearn import linear_model
import math

def translate(letra):
    mapa_numeros = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"sixe":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11}
    return mapa_numeros[letra]

df = pd.read_csv('Hoja_pandas.csv')



df.experience = df.experience.fillna("zero")

for i in range(len(df.experience)):
    df.experience[i] = translate(df.experience[i])
median_nan = math.floor(df.test_score.median())
print(df.test_score.median())
df.test_score = df.test_score.fillna(median_nan)

print(df)

reg = linear_model.LinearRegression()
reg.fit(df[["experience","test_score","interview_score"]],df.salary)
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))



















"""
df = pd.read_csv('price.csv', skiprows=1)#Me confundi al escribri el csv, deje la primera fila vacia

median_bedrooms = math.floor(df.bedrooms.median())
df.bedrooms = df.bedrooms.fillna(median_bedrooms)#Sustituye los NAN de la serie por el atributo que introduzcas
print(df)
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.coef_)
print(reg.predict([[3000,3,40]]))
print(reg.predict([[2500,4,5]]))
"""