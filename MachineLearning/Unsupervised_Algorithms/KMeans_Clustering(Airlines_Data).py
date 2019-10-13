#!/usr/bin/env python
# coding: utf-8

# # KMEANS CLUSTERING

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv(f"/home/prajakta/Downloads/AirlinesCluster.csv")
dataset


# In[3]:


dataset.columns


# # in clustering it is compulsory that y is not present as it is unsupervised learning

# In[4]:


#we will find distance between points with the help of euclidean and manhattan distance
#here accuracy is not goal cluster is.
#acc to y i.e no of clusters we decide our k


# In[5]:


dataset1=dataset#duplicate dataset as if we want to modify it will modify in dataset1 and dataset will remain as it is
dataset1.head()


# In[6]:


dataset1.describe()


# In[7]:


dataset1.info()


# In[8]:


#all the values are in numerical and there are no null values
#no feature selection as we require all the values of x in clustering


# In[9]:


#scaling->scale dataset to normal distribution as otherwise u will have skewed graph
from sklearn import preprocessing
dataset1_preprocessed = preprocessing.scale(dataset1)
dataset1_preprocessed = pd.DataFrame(dataset1_preprocessed)#type cast in data frame


# In[10]:


#Find the appropriate cluster number i.e value of k by ELBOW METHOD
plt.figure(figsize=(12,8))
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++',random_state=42)
    kmeans.fit(dataset1_preprocessed)
    wcss.append(kmeans.inertia_)#for plotting
print(wcss)


# In[11]:


#plotting
plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD FOR K")
plt.xlabel("No of clusters")
plt.ylabel("kmeans_inertia_")


# In[12]:


#k means algorithm
#fitting the k-means to the dataset
kmeans = KMeans(n_clusters=5, init = 'k-means++',random_state=42)
kmeans.fit(dataset1_preprocessed)
y_kmeans = kmeans.fit_predict(dataset1_preprocessed)
#beginning of the cluster numbering with 1 instead of 0
y_kmeans
y_kmeans1 = y_kmeans+1#if we dont add 1 then the cluster will start from 0 and it will be wrong
#New dataframe called cluster
cluster = pd.DataFrame(y_kmeans1)
#adding cluster to the dataset1
dataset1['cluster'] = cluster
#means of clusters
kmeans_mean_cluster = pd.DataFrame(round(dataset1.groupby('cluster').mean(),1))
kmeans_mean_cluster


# In[13]:


#here we have mean values of balance , qualmiles,etc
#here all cluster
kmeans.predict([[152724.4 ,77.9 ,50999.4 ,21.3 ,479.4 ,1.5 ,4912.2]])


# In[14]:


kmeans.n_clusters#no of clusters


# In[15]:


print(kmeans.cluster_centers_)#centroid values


# # HIERARCHICAL CLUSTERING

# *This is a hierarchical clustering

# In[16]:


from sklearn import preprocessing
dataset2_preprocessed = preprocessing.scale(dataset1)
dataset2_preprocessed = pd.DataFrame(dataset1_preprocessed)#type cast in data frame


# In[17]:


#dendogram is used for hierarchical clustering(top down)
#dendogram-> 
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=5, suppress=True)
#supress is scientific float notation
#creating the linkage matrix
#method =ward uses ward avriance minimization algorithm.
H_cluster = linkage(dataset2_preprocessed,method='ward')
plt.figure(figsize=(12,8))
plt.title('Hierarchical Clustering DEndogram(trucated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
#H_cluster ->linkage N-D array of histogram
#'lastp' ->non singleton clusters formed in the linkage are the only non-leaf nodes in the linkage
#level - No more than p levels of the dendogram tree are displayed
dendrogram(
          H_cluster,
          truncate_mode='lastp',#show only the last p merged clusters
          p=5, #show only the last p merged clusters
          leaf_rotation=90,
          leaf_font_size=12.,
          show_contracted=True
)


# In[18]:


#Assigning the clusters and plotting the observations as per hierarchical
from scipy.cluster.hierarchy import fcluster
k=5
cluster_2 = fcluster(H_cluster,k,criterion='maxclust')
cluster_2[0:30:,]
plt.figure(figsize=(12,8))
plt.scatter(dataset2_preprocessed.iloc[:,0],
           dataset2_preprocessed.iloc[:,1],
           c=cluster_2, cmap='prism')
#plot points with cluster dependent colors
plt.title("Airline Data-Hierarchical clustering")
plt.show()


# In[ ]:




