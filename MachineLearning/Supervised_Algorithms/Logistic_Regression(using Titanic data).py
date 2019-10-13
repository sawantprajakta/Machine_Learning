#!/usr/bin/env python
# coding: utf-8

# # LOGISTIC REGRESSION - TITANIC DATA

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


#read dataset
df = pd.read_csv("/home/prajakta/Downloads/Titanic_Data.csv")
df


# In[3]:


df.head()


# In[4]:


df.count()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='plasma')


# In[8]:


sns.heatmap(df.isnull(),cbar=False,cmap='Pastel1')#to print values


# In[15]:


sns.set_style("darkgrid")
sns.countplot(x="Survived",hue = "Sex",data=df)
plt.show


# #distance plot is used for distribution acc to age
# sns.distplot(df['Age'].dropna(),kde=False,bins=30,color='Purple')
# plt.show()

# # CLEANING AND PREPROCESSING

# In[9]:


plt.figure(figsize=(12,7))
plt.title('Pclass Vs Age Box PLot')
sns.boxplot(x='Pclass',y='Age',data = df, palette = 'PuBu')
plt.show()


# In[21]:


df = df.dropna()
df


# In[22]:


df.info()


# In[ ]:


#i cannot do drop na as we can see our rows became 183


# In[10]:


class_1 = df[df['Pclass']==1]
class_1=class_1.dropna()
sum_age = class_1['Age'].sum()
avg_age = sum_age/class_1.count()
avg_age


# In[12]:


class_1 = df[df['Pclass']==2]
class_1=class_1.dropna()
sum_age = class_1['Age'].sum()
avg_age = sum_age/class_1.count()
avg_age


# In[13]:


class_1 = df[df['Pclass']==3]
class_1=class_1.dropna()
sum_age = class_1['Age'].sum()
avg_age = sum_age/class_1.count()
avg_age


# In[34]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 26
        else:
            return 21
    else:
        return Age
    
df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)


# In[15]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[17]:


df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
df.info()


# In[18]:


sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)
embark


# In[19]:


#drop the sex,embarked,name and tickets columns
df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace = True)


# In[20]:


#concentrate on new sex and embark column to our train dataframe bcoz we want the binary and odinal data set numerical
df = pd.concat([df,sex,embark],axis=1)
df.head()#check head of data frame


# In[21]:


#split the dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop("Survived",axis = 1),df['Survived'],test_size=0.20,random_state=101)


# In[26]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=30000)
lr.fit(x_train,y_train)


# In[27]:


accuracy = lr.score(x_test,y_test)
accuracy


# In[ ]:


##############################solvers##########################################
#liblinear - for small dataset
#ibfgs - faster for multinomial class


# In[28]:


#prediction;-1-survived, 0->unsurvived
prediction = lr.predict([[5,3,35.0,0,0,8.0500,1,0,1]])
prediction


# In[32]:


prediction = lr.predict(x_test)
prediction


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# In[ ]:




