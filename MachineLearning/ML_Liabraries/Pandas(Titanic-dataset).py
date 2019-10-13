#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df = pd.read_csv("E:\ACDS\Sem_2\ML\lab\datasets\Titanic_Data.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.Age.head() #df.Sex


# In[8]:


Survived_ = df[df['Survived']==1]
len(Survived_)


# In[9]:


male = df[df['Sex']=='male']
female = df[df['Sex']=='female']


# In[10]:


male.head()


female.head()


# 
# 
# 
# 

# In[11]:


male_survived = male[male['Survived']==1]
female_survived = female[female['Survived']==1]


# In[12]:


female_survived


# In[13]:


a= len(male_survived)/len(Survived_)


# In[14]:


a


# In[15]:


b= len(female_survived)/len(Survived_)
b


# In[ ]:




