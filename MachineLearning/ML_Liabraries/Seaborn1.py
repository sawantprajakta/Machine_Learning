#!/usr/bin/env python
# coding: utf-8

# # Seaborn-part1

# In[9]:


import seaborn as sns#same as matplotlib but this makes complex dataset easy and if u dont have complex data use matplotlib


# In[10]:


import matplotlib.pyplot as plt


# In[16]:


#load dataset tips which is already inbuilt
tips_val= sns.load_dataset('tips')
tips_val


# In[17]:


tips.info()#to check if any null values are present or not


# In[18]:


tips.dropna()#to drop na values


# In[19]:


#to see describtion
tips.describe()


# In[21]:


#to plot 

sns.set_style('darkgrid')
sns.violinplot(x = 'total_bill',data = tips_val)#pass the column to the x ehich you want graph and pass dataset
plt.title("Total Bill")


# In[22]:


#new inbuilt dataset iris which is categorial data
iris_val = sns.load_dataset("iris")
iris_val


# In[24]:


iris_val['species']


# In[26]:


sns.swarmplot(x="species",y = "petal_length",data = iris_val)
plt.show()


# In[27]:


titanic_val=sns.load_dataset("titanic")
titanic_val


# In[28]:


titanic_val.info()


# In[ ]:


#we cant apply drop na if there are more values which are na bcoz it will make our rows less and as here many values are na


# In[32]:


#plt the graph using factor plot
t = sns.catplot("class","survived","sex",data = titanic_val,kind = "bar",palette="muted")#here we pass x,y,category,kind of plotand legend..legend is bydefault true

plt.show()


# In[33]:


t1 = sns.catplot("day","total_bill","smoker",data = tips_val,kind = "bar",palette="muted")#here we pass x,y,category,kind of plotand legend..legend is bydefault true

plt.show()


# In[34]:


#relational plot
sns.relplot(x="total_bill",y="tip",hue="smoker",style="time",data = tips_val)
plt.show()


# In[39]:


sns.relplot(x="petal_length",y="sepal_length",hue="species",style="species",data = iris_val)
plt.show()


# In[42]:


j = sns.relplot(x="sepal_length",y="sepal_width",hue="species",style="species",data=iris_val)
j.set_title("IRIS DATA")
plt.show()


# In[43]:


#BOX PLOTTING
tip = sns.load_dataset("tips")
tip
bx = sns.boxplot(x="total_bill",data = tip)
bx.set_title("Boxplot")


# In[48]:


import numpy as np
import pandas as pd
x = 10**np.arange(1,10)#we generate x and y value
y = x*2

data = pd.DataFrame(data={'x':x,'y':y})#we want to create data now

#plotting
lm = sns.lmplot(x = "x",y = "y",data= data,size = 7,scatter_kws = {"s":100})

lm.set_xticklabels(rotation = 90)

plt.show()


# In[49]:


sns.relplot(x="total_bill",y="tip",hue="size",style="time",size = "size",data = tips_val)
plt.show()


# In[50]:


sns.relplot(x="fare",y="age",hue="alive",size = "sex",data = titanic_val)
plt.show()


# In[ ]:




