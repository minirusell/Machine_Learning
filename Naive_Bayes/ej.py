import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis = 'columns',inplace=True)
dummies = pd.get_dummies(df.Sex,dtype=int)
df = pd.concat([df,dummies],axis='columns')
df.drop('Sex',axis='columns',inplace=True)
df.columns[df.isna().any()]#Si hay algun NAN
df.Age = df.Age.fillna(df.Age.median())

df1 = df[df.Survived == 1]
figure, axes = plt.subplots(2,2)
axes[0,0].hist(df1.Pclass, bins = 10, edgecolor = 'black')
axes[0,0].set_xlabel('Pclass')
axes[0,0].set_ylabel('Frecuencia')
axes[1,0].hist(df1.Age, bins = 10, edgecolor = 'black')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Frecuencia')
axes[0,1].hist(df1.Fare, bins = 10, edgecolor = 'black')
axes[0,1].set_xlabel('Fare')
axes[0,1].set_ylabel('Frecuencia')
axes[1,1].hist(df1.male, bins = 10, edgecolor = 'black')
axes[1,1].set_xlabel('male')
axes[1,1].set_ylabel('Frecuencia')
plt.show()

edad_bins, edad_bordes = np.histogram(df1.Age, bins=10)
fare_bins, fare_bordes = np.histogram(df1.Fare, bins=10)
pclass_bins, pclass_bordes = np.histogram(df1.Pclass, bins=10)
male_bins, male_bordes = np.histogram(df1.male,bins=10)
df.male.eq(0).sum()
P_hombre = df.male.eq(1).sum()/len(df.male)
P_mujer =df.male.eq(0).sum()/len(df.male)
Tot_edad, _ = np.histogram(df.Age, bins=10)
P_edad = Tot_edad/ Tot_edad.sum()
Tot_tarifa,_ = np.histogram(df.Fare, bins=10)
P_tarifa = Tot_tarifa / Tot_tarifa.sum()
Tot_pclass,_ = np.histogram(df.Pclass, bins=10)
P_pclass = Tot_pclass / Tot_pclass.sum()
