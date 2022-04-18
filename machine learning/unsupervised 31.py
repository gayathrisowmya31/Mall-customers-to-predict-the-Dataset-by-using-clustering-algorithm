# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:40:12 2022

@author: Visitor
"""

# unsupervised learning
# we do have data Dv
#k-mean clustering
#import numpy for numeric caluculations
#import pandas for Data manipulation
#import matplot.lib for Data visulization
#Elbow curve method for caluculate average distance to the centroid across all data points
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the data set
data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values
# using the elbow method to find the optimal number
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('The number od cluster')  
plt.ylabel('wcss')
plt.show()
kmeans=KMeans(n_clusters=5,random_state=42)
kmeans.fit(x)
ymeans=kmeans.fit_predict(x)
# visualising the cluster
plt.scatter(x[ymeans==0,0],x[ymeans==0,1],s=100,c='red',label='cluster 1')
plt.scatter(x[ymeans==1,0],x[ymeans==1,1],s=100,c='blue',label='cluster 2')
plt.scatter(x[ymeans==2,0],x[ymeans==2,1],s=100,c='green',label='cluster 3')
plt.scatter(x[ymeans==3,0],x[ymeans==3,1],s=100,c='cyan',label='cluster 4')
plt.scatter(x[ymeans==4,0],x[ymeans==4,1],s=100,c='brown',label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='centroids')
plt.title('Grouped by shopping scores')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()
