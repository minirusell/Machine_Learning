import pandas as pd
import math
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('titanic.csv')
inputs = df.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis = 'columns')
target = df.Survived
##Variables ficticias, considero que sexo es una categoria nominal
dummies = pd.get_dummies(inputs['Sex'])
merged = pd.concat([inputs,dummies], axis = 'columns')
merged = merged.drop(['Sex','male'], axis = 'columns')
median_nan_Age = math.floor(merged.Age.median())
median_nan_Fare = math.floor(merged.Fare.median())
merged.Age = merged.Age.fillna(median_nan_Age)
merged.Fare = merged.Fare.fillna(median_nan_Fare)

## Considero que sexo es una categoria ordinal
le_Sex = LabelEncoder()
inputs['Sex_n'] = le_Sex.fit_transform(inputs['Sex'])
inputs_n = inputs.drop('Sex',axis = 'columns')

X_train1, X_test1, y_train1, y_test1 = train_test_split(merged,target,train_size=0.2)
X_train2, X_test2, y_train2, y_test2 = train_test_split(inputs_n,target,train_size=0.2)


model11 = tree.DecisionTreeClassifier(criterion='gini')
model12 = tree.DecisionTreeClassifier(criterion= 'entropy')
model13 = tree.DecisionTreeClassifier(criterion= 'log_loss')
model21 = tree.DecisionTreeClassifier(criterion='gini')
model22 = tree.DecisionTreeClassifier(criterion= 'entropy')
model23 = tree.DecisionTreeClassifier(criterion= 'log_loss')

model11.fit(X_train1,y_train1)
model12.fit(X_train1,y_train1)
model13.fit(X_train1,y_train1)
model21.fit(X_train2,y_train2)
model22.fit(X_train2,y_train2)
model23.fit(X_train2,y_train2)

print(model11.score(X_test1,y_test1))
print(model12.score(X_test1,y_test1))
print(model13.score(X_test1,y_test1))
print(model21.score(X_test2,y_test2))
print(model22.score(X_test2,y_test2))
print(model23.score(X_test2,y_test2))




