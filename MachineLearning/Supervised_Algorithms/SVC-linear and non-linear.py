#!/usr/bin/env python
# coding: utf-8

# # SUPPORT VECTOR MACHINE

# # SVL WITH MAKE_BLOBS

# In[4]:


#import all necessary libraries
import numpy as np #to gnerate data
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


from sklearn.datasets.samples_generator import make_blobs


# In[24]:


x,y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
sns.set_style('darkgrid')#grey background
plt.scatter(x[:,0],x[:,1],c=y, s=50, cmap='winter')


# In[25]:


#generate data
xfit = np.linspace(-1,3.5)
xfit


# In[26]:


#tp draw line->y=mx+b
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='winter')
plt.plot([0.6],[2.1],'^',color='red',markeredgewidth=2,markersize=10)
#cross->blank, square->s triangle->^
for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit, m*xfit + b, '-k')


# In[27]:


#to find width of margin
xfit = np.linspace(-1,3.5)
xfit
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='winter')
plt.plot([0.6],[2.1],'^',color='red',markeredgewidth=2,markersize=10)
#cross->blank, square->s triangle->^
for m,b,d in [(1,0.65,0.33),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = m*xfit+b
    plt.plot(xfit, m*xfit + b, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor = 'none',color='#AAAAAA', alpha=0.3)


# In[28]:


from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)#cost ->c->if increase hard margin, less ->soft margin,c value=1 by default
#E->2.71 raise to power 10-> for high level value
model.fit(x,y)


# In[39]:


#gca - > to get the current polar axes on current figure ->value of x and y of current figure
#if only x value then it will predict itself for y value and return the values
def plot_svc_decision_function(model, ax=None, plot_support = True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    print(xlim)
    print(ylim)
    
    #create grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    #meshgrid takes 2 para x,y and returns coordinate matrices from coordinate vectors
    #Makes N-D coordinate array from vectorized N-D scalar/vector
    Y,X = np.meshgrid(y,x)
    #Row wise stack in sequence
    xy = np.vstack([X.ravel(),Y.ravel()]).T# T -> transpose
    #SVC - decision function returns pairwise scores between classes-score for each class
    P = model.decision_function(xy).reshape(X.shape)
    
    #plot decision margins and boundaries
    ax.contour(X,Y,P, colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
              
    
    #plot support vectors
    if plot_support:
        #model.support_vectors_-> support vectors coordinates
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=100,linewidth=1,facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    


# In[40]:


plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='winter')
plot_svc_decision_function(model)


# # SVL WITH MAKE_CIRCLES

# In[53]:


from sklearn.datasets.samples_generator import make_circles
X,y = make_circles(100,factor=.1,noise=.1)
clf = SVC(kernel='rbf',C=1E20)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='winter')
plot_svc_decision_function(clf,plot_support=False)


# In[50]:


#radial basis function centered on the middle clump
r = np.exp((-X**2).sum(1))#radius


# In[51]:


#to plot 3D graph
from mpl_toolkits import mplot3d
from ipywidgets import interact,fixed 

def plot_3D(elev=60, azim=30, X=X, y=y):#to draw 3d projection we use elev and azim
    ax=plt.subplot(projection='3d')
    ax.scatter3D(X[:,0],X[:,1],r,c=y, s=50, cmap ='winter')#x1,x2,y,r values passed
    ax.view_init(elev=elev, azim = azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r'),
interact(plot_3D, elev=[0,90],azip=(-360,360),X=fixed(X), y=fixed(y))


# In[ ]:





# In[ ]:




