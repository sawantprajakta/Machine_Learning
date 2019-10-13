#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,auc

from sklearn.metrics import recall_score,precision_score,accuracy_score,f1_score


# In[66]:


df = pd.read_csv("/home/prajakta/Desktop/creditcard.csv")
print(df.shape)
print(df.head(5))


# In[67]:


df.describe()


# In[68]:


#check class variables that contains 0 for genuine and 1 for fraud
fig, ax = plt.subplots(1,1)
ax.pie(df.Class.value_counts(),autopct='%1.1f%%', labels=['Genuine','Fraud'],colors =['yellow','k'])
plt.axis('equal')
plt.ylabel(' ')
plt.show


# In[69]:


#PLot time to see if there is any trend
print("Time variable")
df["Time"]
df['Time_Hr']=df["Time"]/3600
print(df["Time_Hr"].tail(5))


# In[70]:


fig,(ax1,ax2) = plt.subplots(2,1, sharex = True, figsize=(6,3))
ax1.hist(df.Time_Hr[df.Class==0],bins=48, color='pink',alpha=0.5)#for continous data only histogram is made and no bar graph as bar graph is for only discrete value
ax1.set_title("Genuine")
ax2.hist(df.Time_Hr[df.Class==1],bins=48,color='red',alpha=0.5)
ax2.set_title("Fraud")
plt.xlabel("Time(hrs)")
plt.ylabel("Transactions")
plt.show()


# In[71]:


df = df.drop(['Time'],axis=1)


# In[72]:


#CORELATION BETWEEN DATASETS
from sklearn.preprocessing import StandardScaler
df['scaled_Amount']= StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis=1)


# In[73]:


import seaborn as sns
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(28,1)
plt.figure(figsize=(6,28*4))
for i,col in enumerate(df[df.iloc[:,0:28].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(df[col][df.Class == 1],bins = 50, color='blue')
    sns.distplot(df[col][df.Class == 0],bins = 50, color='green')
    ax5.set_xlabel('')
    ax5.set_title('feature:' + str(col))
plt.show()


# In[75]:


from sklearn.naive_bayes import GaussianNB
y = df['Class'].values
X = df.drop(['Class'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 42, stratify = y)
clf_nb = GaussianNB()
clf_nb.fit(X_train,y_train)
y_pred = clf_nb.predict(X_test)
print(y_pred)
y_pred_prob = clf_nb.predict_proba(X_test)
print(y_pred_prob)
Gauss_accuracy = clf_nb.score(X_test,y_test)
print(Gauss_accuracy)


# In[76]:


df1 = pd.read_csv("/home/prajakta/Desktop/creditcard.csv")
print(df1.shape)
print(df1.head(5))


# In[78]:


#to make confusion matrix and to increase accuracy by dropping the same v's IE COLUMNS which we know by corelation graph
y = df1['Class'].values
X = df1.drop(['V8','V13','V15','V20','V28','V27','V26','V25','V24','V23','V22'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2,random_state = 42, stratify = y)
clf_nb = GaussianNB()
clf_nb.fit(X_train,y_train)
y_pred = clf_nb.predict(X_test)
print(y_pred)
y_pred_prob = clf_nb.predict_proba(X_test)
print(y_pred_prob)
Gauss_accuracy = clf_nb.score(X_test,y_test)
print(Gauss_accuracy)
A =confusion_matrix(y_test,y_pred)
A


# In[ ]:




