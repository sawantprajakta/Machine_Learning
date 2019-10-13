#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
col_names = ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
wine = pd.read_csv(url, names=col_names)
wine


# In[2]:


wine.info()


# In[3]:


x_train,x_test,y_train,y_test =train_test_split(wine.drop('Class',axis=1),wine['Class'],test_size=0.1,random_state=101)

logit = LogisticRegression(C=1E20,solver='lbfgs',max_iter=90000,multi_class='multinomial')
logit.fit(x_train,y_train)
confidence = logit.score(x_test,y_test)
print(confidence)
prediction = logit.predict(x_test)


# In[4]:


from sklearn.svm import SVC
model = SVC(C=1E20,kernel='rbf',gamma="scale",decision_function_shape='ovo')#cost ->c->if increase hard margin, less ->soft margin,c value=1 by default
model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(accuracy)


# In[ ]:




