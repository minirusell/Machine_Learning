import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('homeprices.csv')
dummies = pd.get_dummies(df.town)#Te crea variables fictiacias,
print(dummies)



merged = pd.concat([df,dummies],axis= 'columns')
final = merged.drop(['town','west windsor'],axis='columns')#Se eliminan las categorias que quieras
model = linear_model.LinearRegression()
X = final.drop('price',axis='columns')
y = final.price

print(X)
print(y)
model.fit(X,y)
print(model.predict([[2800,False,True]]))
print(model.predict([[3400,False,False]]))
print(model.score(X,y))#Que tan preciso es el modelo de entrenamiento

##########################################################################
le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)#Toma la columna y la transforma a etiquetas
X = dfle[['town','area']].values
y = dfle.price
ohe = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(),[0])],
    remainder= 'passthrough'
)
X = ohe.fit_transform(X)
print(X)
X = X[:,1:]
model.fit(X,y)
print(model.predict([[1,0,2800]])) 