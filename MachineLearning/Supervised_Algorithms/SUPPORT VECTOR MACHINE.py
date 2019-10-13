#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()


# In[48]:


from sklearn.datasets.samples_generator import make_blobs
X,y = make_blobs(n_samples=50, centers = 2,
                random_state=0, cluster_std=0.60)
plt.scatter(X[:,0],X[:, 1], c=y, s=50, cmap='autumn')#c for color, s for samples, cmap for background


# In[49]:


#how we draw random line or boundary line putting in eqation mx+c
xfit = np.linspace(-1, 3.5)
xfit
plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='autumn')
plt.plot([0.6],[2.1],'x', color='red', markeredgewidth=2, markersize=10)
for m, b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:#values of x & y
    plt.plot(xfit, m*xfit + b, '-k')#mx+c we have drawn lines with random points
plt.xlim(-1,3.5)


# In[53]:


xfit = np.linspace(-1,3.5)
plt.scatter(X[:,0], X[:,1], c =y, s=50, cmap = 'autumn')
for m, b, d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:#values of x & b & d
    yfit = m*xfit + b#mx+c we have drawn lines with random points
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit, yfit - d, edgecolor='none',
                    color='#AAAAAA',alpha=0.4)


# In[54]:


from sklearn.svm import SVC #Support Vector Classifier
model = SVC(kernel='linear', C=1E10)
model.fit(X,y)


# In[60]:


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    #create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margines
    ax.contour(X,Y,P, colors ='k',
              levels=[-1,0,1], alpha=0.5,
              linestyles=['--','-','--'])
    #plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],
                  model.support_vectors_[:,1],
                  s=300,linewidth=1,facecolors='none')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    


# In[61]:


plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);


# In[ ]:





# In[ ]:




