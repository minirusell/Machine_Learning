import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('carprices.csv')

figure, axes = plt.subplots(2,1)
axes[0].scatter(df['Mileage'],df['Sell Price($)'],color = 'red', marker='+')
axes[1].scatter(df['Age(yrs)'],df['Sell Price($)'],color = 'blue')
axes[0].set_xlabel('Mileage')
axes[0].set_ylabel('Sell Price($)')
axes[1].set_xlabel('Age(yrs)')
axes[1].set_ylabel('Sell Price($)')
plt.show()

X = df[['Mileage','Sell Price($)']]
y = df['Sell Price($)']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)#Elige las muestras aleatorias
model = linear_model.LinearRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))
print(y_test)
print(model.score(X_test,y_test))