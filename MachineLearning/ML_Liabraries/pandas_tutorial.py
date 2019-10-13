#!/usr/bin/env python
# coding: utf-8

# ## PANDAS
# Importing Pandas
# In[1]:


import numpy as np
import pandas as pd


# # Series
# 
# pd.Series(data,index)

# In[2]:


Student = pd.Series(data=["Alex","Diana","Bob","Parth","Ema"])
print(Student)

print(Student.index)


# In[3]:


Student_index=pd.Series(data=["Alex","Diana","Bob","Parth","Ema"],index=['101','102','103','104','105'])
Student_index


# # Create a Series object from python list :

# In[4]:


Stud =["Niki","Sam","Paul"]
print(Stud)


# In[5]:


print(pd.Series(Stud))


# # Create a Series object from python dictionary :

# In[6]:


Phonebook = { 
              "Alex" : 9011786101,
              "Diana" : 9822343043,
              "Parth" : 9850883043
            }
print(Phonebook)


# In[7]:


p=pd.Series(Phonebook)
p


# In[8]:


# Accessing the elements in Series
Phonebook["Parth"]


# In[9]:


p.values


# In[10]:


p.index


# # DataFrames

# In[11]:


movie_details = pd.DataFrame({
         "movie_id":[101,102,103,104,105,106],
         "movie_name":["Smurfs","Harry-Potter","Minions","DeadPool","Tinker-Bell","Ted2"],
         "movie_rating":[6,8,9,7,5,9]
        })
movie_details


# # Rename

# In[12]:


movie=movie_details.rename(columns={'movie_name':'Movies','movie_id':'Movie_id','movie_rating':'Ratings'})
movie


# # OPERATIONS ON DATAFRAMES

# # 1.Slicing

# In[13]:


print(movie.head(2))


# In[14]:


print(movie.tail(2))


# # 2.Concatenation

# In[15]:


movie1 =pd.DataFrame({"Movie_id":[107,108],
         "Movies":["Fault in our Star","Spider-Man3"],
         "Ratings":[6,9]
        })
movie1


# In[16]:


print(movie,"\n\n" , movie1)

#Concatinate the two dataframes along rows:
# In[17]:


con=pd.concat([movie,movie1])
con

#Concatinate the two dataframes along columns:
# In[18]:


con1=pd.concat([movie, movie1], axis=1)
con1


# # 3.Merging and Joining

# In[19]:


movie2=pd.DataFrame({
         "Movie_id":[107,108],
         "Movies":["Fault in our Star","Spider-Man3"],
         "Category":["Drama|Romance","Thriller|Children"]
        })
movie2


# In[20]:


merging = pd.merge(movie1,movie2,on="Movie_id")
merging


# #Joining:

# In[21]:


outer_join=pd.merge(movie,movie1,on="Movie_id",how="outer")
outer_join


# In[22]:


inner_join=pd.merge(movie1,movie2,on="Movie_id",how="inner")
inner_join


# # 4.Indexing 
# 
# What are indexes for?
# 
# 1.Identification
# 2.Selection
# 

# In[23]:


movie_details.index


# 1.Identification

# In[24]:


movie[movie.Ratings==9]


# 2.Selection

# In[25]:


##select all rows for a specific column
movie.loc[:,'Movies']


# In[26]:


print(movie.iloc[[4], [1, 2]],"\n\n" , movie)


# In[27]:


# inplace=True makes the change 
# sets the index to 'country'
movie.set_index('Movie_id', inplace=True)
movie


# In[28]:


print("Index:", movie.index)
print("Columns :", movie.columns)


# In[29]:


## say you prefer to use the default index and you want back the column

movie.index.name = 'Movie_id'
movie.reset_index(inplace=True)
movie


# # 5. Load data from various file formats

#  CSV

# In[30]:


df = pd.read_csv('E:\Datasets\Titanic_Data.csv')
df.head()


# In[31]:


import sqlite3
con = sqlite3.connect("mysql.db")


# In[ ]:





# # 6. Data Munging
# In Data munging, you can convert a particular data into a different format. For example, 
# if you have a .csv file, you can convert it into .html or any other data format as well. 
# So, let me implement this practically.

# In[32]:


df2= pd.read_csv('E:\Datasets\Titanic_Data.csv')
 
df2.to_html('E:\Datasets\edu.html')


# # Data Clean Up using Pandas
#    
#    While doing machine learning problems, most of the times, the data that is available may not be clean and perfect. There may be missing values, unwanted data, and a lot of problems. 
# So, it is very important to clean the data before we use it for machine learning purposes. Let’s see some ways by which we can clean the data in pandas.

# In[33]:


Patient = pd.read_csv('E:\Datasets\Diabetes_Data.csv')
Patient


# #Dropping Null Value Rows

# In[34]:


Patient.dropna()


# #Filling Null Values by Some Other Values

# In[35]:


Patient.fillna(0)


# In[36]:


v = {'BloodPressure': 1,'Insulin':2}
Patient.fillna(value=v)


# #Grouping of Data

# In[ ]:





# In[37]:


Patient.groupby('Name').mean()


# In[38]:


Patient.drop(["BMI", "DiabetesPedigreeFunction"],axis = 1,  inplace = True)
Patient


# In[ ]:




