import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('income.csv')

plt.scatter(df['Age'],df['Income($)'],color = 'red', marker = '.')
plt.show()



scalar = MinMaxScaler()
scalar.fit(df[['Age']])
df['Age'] = scalar.transform(df[['Age']])
scalar.fit(df[['Income($)']])
df['Income($)'] = scalar.transform(df[['Income($)']])
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print(y_predicted)
df['cluter'] = y_predicted


df1 = df[df.cluter==0]
df2 = df[df.cluter==1]
df3 = df[df.cluter==2]
plt.scatter(df1.Age,df1['Income($)'],color = 'red', marker = '.')
plt.scatter(df2.Age,df2['Income($)'],color = 'blue', marker = '+')
plt.scatter(df3.Age,df3['Income($)'],color = 'black', marker = '*')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color ='purple',marker='^')
plt.xlabel('Age'),
plt.ylabel('Income($)')
plt.show()

k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()



