import numpy as np

def gradient_descent(x,y):
    m_act = b_act = 0
    interations = 10
    n = len(x)
    learning_rate = 0.0015
    for i in range(interations):
        y_predicted = m_act * x + b_act
        cost = (1/n)*sum((y-y_predicted)**2)
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_act = m_act - learning_rate * md
        b_act = b_act - learning_rate * bd
        print(f"m {m_act}, b {b_act}, cost {cost} ,interation {i}")




x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)
