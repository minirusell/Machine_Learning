import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('insurance_data.csv')
plt.scatter(df.age ,df.bought_insurance ,color = 'red', marker='+')
plt.xlabel('age')
plt.ylabel('bought_insurance')
plt.show()
X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance, test_size=0.1)
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test, y_test)

