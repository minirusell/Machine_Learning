import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
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

X_train, X_test, y_train, y_test = train_test_split(df.drop(['Survived','female'], axis='columns'),df.Survived,test_size=0.2)

train_df = X_train.copy()
train_df["Survived"] = y_train.values

X_trainS  = train_df[train_df.Survived == 1]
X_trainNS = train_df[train_df.Survived == 0]

#Probabilidades condicionadas
_, edad_bordes = np.histogram(X_train.Age, bins=n)
_, fare_bordes = np.histogram(X_train.Fare, bins=n)

    #df.Pclass

Smale_aux = np.array(X_trainS.male.array)  
Smale_bins = np.array([np.sum(Smale_aux == 0),np.sum(Smale_aux == 1)])
SPclass_aux = np.array(X_trainS.Pclass.array)  
SPclass_bins = np.array([np.sum(SPclass_aux == 1),np.sum(SPclass_aux == 2),np.sum(SPclass_aux == 3)])

NSmale_aux = np.array(X_trainNS.male.array)  
NSmale_bins = np.array([np.sum(NSmale_aux == 0),np.sum(NSmale_aux == 1)])
NSPclass_aux = np.array(X_trainNS.Pclass.array)  
NSPclass_bins = np.array([np.sum(NSPclass_aux == 1),np.sum(NSPclass_aux == 2),np.sum(NSPclass_aux == 3)])


Sedad_bins, _ = np.histogram(X_trainS.Age, bins=edad_bordes)
Sfare_bins, _ = np.histogram(X_trainS.Fare, bins=fare_bordes)
NSedad_bins, _ = np.histogram(X_trainNS.Age, bins=edad_bordes)
NSfare_bins, _ = np.histogram(X_trainNS.Fare, bins=fare_bordes)


#suavizado de 0
alpha = 1.0

PCS_edad = (Sedad_bins + alpha)/(Sedad_bins.sum()+alpha*n)
PCS_fare = (Sfare_bins + alpha)/(Sfare_bins.sum() + alpha*n)
PCS_pclass = (SPclass_bins + alpha)/(SPclass_bins.sum() + alpha*3)
PCS_male = (Smale_bins + alpha)/(Smale_bins.sum() + 3*alpha)

PCNS_edad = (NSedad_bins + alpha)/(NSedad_bins.sum() +  alpha*n)
PCNS_fare = (NSfare_bins + alpha)/(NSfare_bins.sum() + alpha*n)
PCNS_pclass = (NSPclass_bins + alpha)/(NSPclass_bins.sum() + alpha*3)
PCNS_male = (NSmale_bins + alpha)/(NSmale_bins.sum() +  alpha*2)

#Probabilidades totales

P_sobrevivir = len(X_trainS.Survived)/len(y_train)
P_Nsobrevivir = 1-len(X_trainS.Survived)/len(y_train)
print(X_trainS.head(1))
# Guardar los valores reales antes de eliminar la columna Survived
y_real = np.array(y_test)



Pclass = np.array(X_test.Pclass.array)
Age = np.array(X_test.Age.array)
Fare = np.array(X_test.Fare.array)
male = np.array(X_test.male.array)
c_index = np.clip(np.digitize(Pclass, [1,2,3]) -1,0, 2)
a_index = np.clip(np.digitize(Age, edad_bordes) - 1, 0, n-1)
f_index = np.clip(np.digitize(Fare, fare_bordes) -1, 0, n-1)
m_index = np.clip(np.digitize(male, [0,1]) -1, 0, 1)





PS_posteriori = PCS_edad[a_index] * PCS_fare[f_index] * PCS_pclass[c_index] * PCS_male[m_index] * P_sobrevivir
PNS_posteriori = PCNS_edad[a_index] * PCNS_fare[f_index] * PCNS_pclass[c_index] * PCNS_male[m_index] * P_Nsobrevivir

y = np.where(PS_posteriori >= PNS_posteriori, 1, 0)

# Calcular la precisión (accuracy)
accuracy = np.mean(y == y_real)
print(f"Precisión (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")


model = GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))

