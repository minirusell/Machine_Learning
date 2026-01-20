import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
import joblib

df = pd.read_csv('Datos_entrenamiento.csv')
model = linear_model.LinearRegression()
model.fit(df[['area']],df.price)
with open('model_pickle','wb') as f:#wb se escribe datos binarios
    pickle.dump(model,f)

with open('model_pickle','rb') as f:#Abres un archivo en modo lectura binaria
    mp = pickle.load(f)

joblib.dump(model, 'model_joblib')
mj = joblib.load('model_joblib')

