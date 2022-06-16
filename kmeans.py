import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

#random list 
randomlist = []
# filling the list with random values
for i in range(0,10):
    n = random.randint(1,100)
    randomlist.append(n)

# converting 1-D list to a numpy array
nplist = np.reshape(randomlist, (10, 1))
# clusters to be made
kmeans = KMeans(2)
# using object to of kmean to learn the model
kmeans.fit(nplist)
# predicting the given values
identified_clusters = kmeans.fit_predict([[32], [97]])
#printing clusters
print(identified_clusters)