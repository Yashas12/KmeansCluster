import pandas as pd
import numpy as np
import matplotlib.pyplot as pp

data = pd.read_csv("C:/Users/admin/Downloads/mc.csv")
#print(data)

x = data.iloc[:,[-2,-1]].values
#print(x)

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init='k-means++', max_iter=300,n_init=10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)

#print(wcss)

'''pp.plot(range(1,11),wcss)
pp.title("Elbow graph for k-means algorithm")
pp.xlabel("K")
pp.ylabel("Variance")
pp.show()'''

km = KMeans(n_clusters=5,init='k-means++', max_iter=300,n_init=10,random_state=0)

y_km=km.fit_predict(x)

pp.scatter(x[y_km==0,0],x[y_km==0,1],s=100,c="r",label="c1")
pp.scatter(x[y_km==1,0],x[y_km==1,1],s=100,c="blue",label="c2")
pp.scatter(x[y_km==2,0],x[y_km==2,1],s=100,c="black",label="c3")
pp.scatter(x[y_km==3,0],x[y_km==3,1],s=100,c="green",label="c4")
pp.scatter(x[y_km==4,0],x[y_km==4,1],s=100,c="magenta",label="c5")

pp.xlabel("salary")
pp.ylabel("score")

pp.legend()

pp.show()


