#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:53:23 2024

@author: williamsommers 
"""

# William Sommers
# HiveMQ Technical Account Manager (TAM)

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# read the data - replace this with MQTT data
df = pd.read_csv("motor_data.csv")
print(df.describe())

S=np.array(list(df['signal']))

# plot a sample of the motor current signal
df.iloc[3000:3200,1].plot.line(x='time', y='signal')
plt.title('Motor Current Signal Sample')
plt.show()


# provide an array of ones (1) for the single-dimension trasformation
ones = []
for i in range(len(S)):
    ones.append(1)


# select 3 clusters of interest (k=3) and run KMeans analysis
k=3
kmeans = KMeans(n_clusters=k)
kmeans = kmeans.fit(S.reshape(-1,1))

# perform the prediction assigning labels to each data point
labels = kmeans.predict(S.reshape(-1,1))

# plot a sample of the cluster labeling (assignment)
plt.plot(labels[3000:3200])
plt.title('Cluster Labeling Per Data Point (Sample)')
plt.show()

# set the centroids variable for the cluster centers
centroids = kmeans.cluster_centers_

print("centroids")
print(centroids[:,0])

print(labels)
print(centroids)

# color map
c = ['y', 'r', 'b', 'g', 'c', 'm']
colors = [c[i] for i in labels]

#sys.exit(0)


# plot the K-means cluster results
plt.scatter(ones, df['signal'], c=colors, s=18)
plt.boxplot(df['signal'])
plt.title('K-Means Clusters')
plt.show()



# run a prediction for a single data point to see which cluster it is assigned
cluster = kmeans.predict([[80]])[0]
print('data value= {0} , cluster = {1}, cluster color = {2}'.format(S[80], cluster,  c[cluster]))

