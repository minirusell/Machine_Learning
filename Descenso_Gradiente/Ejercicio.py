import pandas as pd
import numpy as np
import math

def descenso_gradiente(x,y):
    m_act = b_act = 0
    n = len(x)
    interation = 1000
    learning_rate = 0.000202
    for i in range(interation):
        y_prediction = m_act * x + b_act
        cost = (1/n)*sum((y-y_prediction)**2)
        md = -(2/n)*sum(x*(y-y_prediction))
        bd = -(2/n)*sum(y-y_prediction)
        m_act = m_act - learning_rate*md
        b_act = b_act - learning_rate*bd
        print(f"m {m_act}, b {b_act}, cost {cost}, interation {i}")
        if math.isclose(cost,0, rel_tol = 1e-20):
            break



df = pd.read_csv("test_scores.csv")
x = np.array(df.math)     
y = np.array(df.cs)
descenso_gradiente(x,y)
