#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data


# In[1]:


import pandas as pd
import numpy as np
import urllib.request#package for url-> to take data from url
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import theano
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


urlval = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"


# In[3]:


raw_data = urllib.request.urlopen(urlval)
raw_data
dataset = np.loadtxt(raw_data, delimiter=',')


# In[4]:


print(dataset[0])


# In[5]:


X = dataset[:,0:48]
Y = dataset[:,-1]


# In[6]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = .33,random_state = 17)


# In[7]:


BernNB = BernoulliNB(binarize=True)
BernNB.fit(X_train,Y_train)
Bern_accuracy = BernNB.score(X_test,Y_test)
print (Bern_accuracy)


# In[8]:


BernNB1 = GaussianNB(var_smoothing=1e-09)
BernNB1.fit(X_train,Y_train)
Bern_accuracy1 = BernNB1.score(X_test,Y_test)
print (Bern_accuracy1)
y_predict=BernNB1.predict(X_test)
y_predict
print(accuracy_score(Y_test,y_predict))
       


# In[9]:


BernNB2 = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
BernNB2.fit(X_train,Y_train)
Bern_accuracy2 = BernNB2.score(X_test,Y_test)
print (Bern_accuracy2)


# In[ ]:





# In[ ]:




