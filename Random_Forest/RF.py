from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
digitos = load_digits()
print(dir(digitos))

df = pd.DataFrame(digitos.data,columns=digitos.feature_names)
df['target'] = digitos.target
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis = 'columns'),digitos.target, test_size = 0.2)
model = RandomForestClassifier(n_estimators= 20)
model.fit(X_train, y_train)
print(model.score(X_test,y_test))

y_predicho = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicho)
print(cm)

