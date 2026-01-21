import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('carprices.csv')

figure, axes = plt.subplots(2, 1)

axes[0].scatter(df['Mileage'],df['Sell Price($)'],color = 'red', marker = '+')
axes[1].scatter(df['Age(yrs)'],df['Sell Price($)'],color = 'blue')
axes[0].set_xlabel('Mileage')
axes[0].set_ylabel('Sell Price($)')
axes[1].set_xlabel('Age(yrs)')
axes[1].set_ylabel('Sell Price($)')
plt.show()




dummies = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,dummies],axis='columns')
final = merged.drop(['Car Model','Audi A5'],axis = 'columns')
X = final.drop(['Sell Price($)'],axis = 'columns')
y = final['Sell Price($)']
print(X)
model = linear_model.LinearRegression()
model.fit(X,y)
print(model.score(X,y))
print(model.predict([[45000,4,0,1]]))
print(model.predict([[86000,7,1,0]]))