#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from subprocess import check_output


# In[29]:


myuberdata = pd.read_csv("/home/prajakta/Desktop/My_Uber_drive_2016.csv")
myuberdata


# In[30]:


myuberdata.isnull()
myuberdata.isnull().sum()


# In[31]:


ubercopy=myuberdata.copy()
ubercopy


# In[32]:


ubercopy=ubercopy.drop(ubercopy.index[1155])
ubercopy.isnull().sum()


# In[33]:


start_list=[i.split(' ') for i in ubercopy['START_DATE*'].tolist()]
start_list


# In[34]:


stop_list=[i.split(' ') for i in ubercopy['END_DATE*'].tolist()]
stop_list


# In[35]:


start=pd.DataFrame(start_list,columns=['Start_Date','Start_Time'])
stop=pd.DataFrame(stop_list,columns=['End_Date','End_Time'])


# In[36]:


start


# In[37]:


stop


# In[38]:


sub_split=ubercopy[['CATEGORY*','START*','STOP*','MILES*','PURPOSE*']]
merge_start_stop=pd.concat([start,stop],axis=1)


# In[39]:


ubercopy=pd.concat([merge_start_stop,sub_split],axis=1)


# In[40]:


ubercopy.head(10)


# In[44]:


#corelation between dist travelled
#no discrete data..data is continous but has some range and divide it into frquency to know how much miles does people travel that is they travel small dist or long
#we wll take diff ranges because if only one travels for 5 miles then there is no analysis on it but if we divide it into ranges then we can analyse that uber is prefered for long or short distance
miledata = ubercopy['MILES*']
mile_range = ["<=5","5-10","10-15","15-20",">20"]
def label(rects):
    for rect in rects:
        height = rect.get_height=()

        plt.text(rect.get_x()+rect.get_width()/2.,1.03*height,'%s' %int(height))
ml_dict=dict()
for item in mile_range:
    ml_dict[item]=0
for mile in ml.dis.values:
    if mile<=5:
        ml_dict["<=5"]+=1
    if mile<=10:
        ml_dict["5-10"]+=1
    if mile<=15:
        ml_dict["10-15"]+=1
    if mile<=20:
        ml_dict["15-20"]+=1
    else:
        ml_dict[">=20"]+=1
ml_dis = pd.Series(ml_dict)
ml_dis.sort_values(inplace=True,ascending = False)
print("Miles:\n", ml_dis)
rects = plt.bar(range(1,len(ml_dis.index)+1),ml_dis.values)
plt.title("Miles")
plt.xlabel("Miles")
plt.ylabel("Frequency")
label(rects)
plt.savefig("/home/prajakta/Desktop")


# In[45]:


ubercopy["PURPOSE*"].value_counts()


# In[47]:


plt.figure(figsize=(15,8))#to define the graph widht or to change the size of graph
sns.countplot(ubercopy["PURPOSE*"])


# In[50]:


plt.figure(figsize=(12,12))#to define the graph widht or to change the size of graph
ubercopy["PURPOSE*"].value_counts()[:11].plot(kind="pie",autopct='%1.1f%%',legend= True)
plt.show()


# In[52]:


#Monthly Rides Analysis
ubercopy["Start_Date"]=pd.to_datetime(ubercopy["Start_Date"],format="%m/%d/%Y")
per_month = ubercopy['Start_Date'].dt.month.value_counts()
per_month=per_month.sort_index()
per_month_mean=per_month.mean()
print("Month Distribute:\n",per_month)


# In[57]:


plt.figure(figsize=(18,12))
sns.countplot(ubercopy["Start_Date"].dt.month)
plt.show()


# In[ ]:




