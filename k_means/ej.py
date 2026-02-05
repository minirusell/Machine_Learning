from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

iris = load_iris()

df = pd.DataFrame(iris.data[:,2:4], columns=iris.feature_names[2:4])
scalar = MinMaxScaler()
scalar.fit(df[['petal length (cm)']])
df['petal length (cm)'] = scalar.transform(df[['petal length (cm)']])
scalar.fit(df[['petal width (cm)']])
df['petal width (cm)'] = scalar.transform(df[['petal width (cm)']])
plt.scatter(df['petal length (cm)'],df['petal width (cm)'],color = 'red')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df)
print(y_predicted)
df['cluster'] = y_predicted
df1 = df[df['cluster'] == 0]
df2 = df[df['cluster'] == 1]
df3 = df[df['cluster'] == 2]

plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color = 'blue')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color = 'red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color = 'black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color ='purple',marker='^')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')
plt.show()

k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal length (cm)','petal width (cm)']])
    sse.append(km.inertia_)
plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()



