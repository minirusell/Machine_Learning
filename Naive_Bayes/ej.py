import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('titanic.csv')
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis = 'columns',inplace=True)
dummies = pd.get_dummies(df.Sex,dtype=int)
df = pd.concat([df,dummies],axis='columns')
df.drop('Sex',axis='columns',inplace=True)
df.columns[df.isna().any()]#Si hay algun NAN
df.Age = df.Age.fillna(df.Age.median())
n = 20
df1 = df[df.Survived == 1]
df2 = df[df.Survived == 0]
figure, axes = plt.subplots(2,2)
axes[0,0].hist(df1.Pclass, bins = n, edgecolor = 'black')
axes[0,0].set_xlabel('Pclass')
axes[0,0].set_ylabel('Frecuencia')
axes[1,0].hist(df1.Age, bins = n, edgecolor = 'black')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Frecuencia')
axes[0,1].hist(df1.Fare, bins = n, edgecolor = 'black')
axes[0,1].set_xlabel('Fare')
axes[0,1].set_ylabel('Frecuencia')
axes[1,1].hist(df1.male, bins = n, edgecolor = 'black')
axes[1,1].set_xlabel('male')
axes[1,1].set_ylabel('Frecuencia')
plt.show()
figure, axes = plt.subplots(2,2)
axes[0,0].hist(df2.Pclass, bins = n, edgecolor = 'black')
axes[0,0].set_xlabel('Pclass')
axes[0,0].set_ylabel('Frecuencia')
axes[1,0].hist(df2.Age, bins = n, edgecolor = 'black')
axes[1,0].set_xlabel('Age')
axes[1,0].set_ylabel('Frecuencia')
axes[0,1].hist(df2.Fare, bins = n, edgecolor = 'black')
axes[0,1].set_xlabel('Fare')
axes[0,1].set_ylabel('Frecuencia')
axes[1,1].hist(df2.male, bins = n, edgecolor = 'black')
axes[1,1].set_xlabel('male')
axes[1,1].set_ylabel('Frecuencia')
plt.show()

#Probabilidades condicionadas
_, edad_bordes = np.histogram(df.Age, bins=n)
_, fare_bordes = np.histogram(df.Fare, bins=n)
_, pclass_bordes = np.histogram(df.Pclass, bins=n)
_, male_bordes = np.histogram(df.male,bins=n)

Sedad_bins, _ = np.histogram(df1.Age, bins=edad_bordes)
Sfare_bins, _ = np.histogram(df1.Fare, bins=fare_bordes)
Spclass_bins, _ = np.histogram(df1.Pclass, bins=pclass_bordes)
Smale_bins, _ = np.histogram(df1.male,bins=male_bordes)
NSedad_bins, _ = np.histogram(df2.Age, bins=edad_bordes)
NSfare_bins, _ = np.histogram(df2.Fare, bins=fare_bordes)
NSpclass_bins, _ = np.histogram(df2.Pclass, bins=pclass_bordes)
NSmale_bins, _ = np.histogram(df2.male,bins=male_bordes)


#suavizado de 0
alpha = 1.0

PCS_edad = (Sedad_bins + alpha)/(Sedad_bins.sum()+alpha*n)
PCS_fare = (Sfare_bins + alpha)/(Sfare_bins.sum() + alpha*n)
PCS_pclass = (Spclass_bins + alpha)/(Spclass_bins.sum() + alpha*n)
PCS_male = (Smale_bins + alpha)/(Smale_bins.sum() + n*alpha)

PCNS_edad = (NSedad_bins + alpha)/(NSedad_bins.sum() +  alpha*n)
PCNS_fare = (NSfare_bins + alpha)/(NSfare_bins.sum() + alpha*n)
PCNS_pclass = (NSpclass_bins + alpha)/(NSpclass_bins.sum() + alpha*n)
PCNS_male = (NSmale_bins + alpha)/(NSmale_bins.sum() +  alpha*n)

#Probabilidades totales

P_sobrevivir = len(df1.Survived)/len(df.Survived)
P_Nsobrevivir = 1-len(df1.Survived)/len(df.Survived)
print(df.head(1))
# Guardar los valores reales antes de eliminar la columna Survived
y_real = np.array(df.Survived.array)
df.drop(['Survived','female'], axis='columns', inplace=True, errors='ignore')


Pclass = np.array(df.Pclass.array)
Age = np.array(df.Age.array)
Fare = np.array(df.Fare.array)
male = np.array(df.male.array)
c_index = np.clip(np.digitize(Pclass, pclass_bordes) -1,0, n-1)
a_index = np.clip(np.digitize(Age, edad_bordes) - 1, 0, n-1)
f_index = np.clip(np.digitize(Fare, fare_bordes) -1, 0, n-1)
m_index = np.clip(np.digitize(male, male_bordes) -1, 0, n-1)





PS_posteriori = PCS_edad[a_index] * PCS_fare[f_index] * PCS_pclass[c_index] * PCS_male[m_index] * P_sobrevivir
PNS_posteriori = PCNS_edad[a_index] * PCNS_fare[f_index] * PCNS_pclass[c_index] * PCNS_male[m_index] * P_Nsobrevivir

y = np.where(PS_posteriori >= PNS_posteriori, 1, 0)

# Calcular la precisión (accuracy)
accuracy = np.mean(y == y_real)
print(f"Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

