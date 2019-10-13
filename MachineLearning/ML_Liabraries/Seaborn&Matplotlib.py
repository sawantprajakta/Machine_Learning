#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot 
import pandas as pd
import seaborn as sns


# In[4]:


df = pd.DataFrame(dict(time=np.arange(500),value=np.random.randn(500).cumsum()))
df
new = sns.relplot(x="time",y="value",kind="line",data=df)#kind for specifing the shape of graph
new.fig.autofmt_xdate()


# In[6]:


#to add uncertainity in graph
#Aggregation & Representation Uncertainity
fmri = sns.load_dataset("fmri")
print(fmri)
sns.relplot(x="timepoint", y="signal", kind="line", data=fmri);


# In[7]:


titanic = sns.load_dataset("titanic")
print(titanic)


# In[8]:


sns.relplot(x="age", y="fare", kind="line", data=titanic);


# In[10]:


sns.relplot(x="age", y="fare", kind="line", ci="sd", data=titanic);


# In[19]:


sns.relplot(x="fare",y="class",hue="survived",data=titanic)#grid graph


# In[23]:


tips = sns.load_dataset("tips")
print(tips)
sns.lineplot(x="total_bill",y="tip",hue="smoker",col="time",data=tips)


# In[ ]:




