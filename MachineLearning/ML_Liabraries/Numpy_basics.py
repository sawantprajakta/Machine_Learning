#!/usr/bin/env python
# coding: utf-8

# # NUMPY BASICS

#  

# In[1]:


import numpy as np


# In[2]:


#Creating an single dimesional array
np_array=np.array([2,3,4,5,6,7,8,9])
print(np_array)


# In[3]:


#to check the type of array
type(np_array)


# # Difference b/w List & NDArray

# In[4]:


#Creating a list
list1=["Pizza","Pizzzaaa","Dominoz Pizza","Topping-Pizza"]
print(list1)


# In[5]:


#we are getting multiple of list1
list2=list1*2
print(list2)


# In[6]:


#in list we get the same list multiple times
python_list=[1,2,3,4,5,6,7,8,9]
print((python_list)*2)


# In[7]:


#in the array we get the product of list
numpy_arr=np.array(([1,2,3,4,5,6]))
print((numpy_arr)*2)


# In[8]:


python_list2=python_list+python_list
print(python_list2)


# In[9]:


#numpy_array=numpy_array + 


# # SOME NUMPY ATTRIBUTES

# In[10]:


#1st method to generate array: 
#Two-dimensional array
numpy_array3=np.array([[2,3,4,5],[4,5,6,7]])
numpy_array3


# In[11]:


#Specify data type-->dtype
numpy_array3=np.array([[2,3,4,5],[4,5,6,7]],dtype=float)
numpy_array3


# In[12]:


#2nd method to generate array
np.arange(1,9,3)


# In[13]:


#3rd method to create the array->to give equal space or divide into equal size values
#parametres: start,stop,step,data type
np.linspace(1,10,5,dtype=float)


# In[14]:


#
np.linspace(1,10,num=8,endpoint=False,dtype=float)
np.linspace(1,10,num=8,dtype=float)


# In[15]:


#numpy matrix
np.arange(1,10).reshape(3,3)


# In[16]:


#zero matrx=ix
np.zeros((1,3))


# In[17]:


#
np.ones((3,3))


# In[18]:


#identical Matrix
np.eye(4)


# In[19]:


#Random Number--->it will generate any random values 
np.random.random(3)


# In[20]:


#Random numbers in form of matrix:
np.random.random((3,5))


# # SLICING

# In[21]:


array_num4=np.random.random((3,5))
array_num4
array_num4[1:4]


# In[22]:


array_num4[0:4,0:4]


# In[23]:


np_array=np.array([2,3,4,5,6,7,8,9])
print(np_array)
np_array[1:4]


# # BROADCASTING
# The term broadcasting describes how numpy treats arrays with different shapes during arithmetic operations  -scipy.org 

# In[24]:


np_val=np.arange(1,10)
np_val


# In[25]:


#Shape-->sizes or number of values in array:
np_val.shape


# In[26]:


np_val1=np.ones(7)
np_val1


# In[27]:


np_val1.shape


# In[28]:


#both arrays are of different size/shape so error
np_val+np_val1


# In[29]:


#Reshape
new_np_val=np_val.reshape(9,1)
new_np_val


# In[30]:


#We get the result after reshaping:
#Sum
new_np_val+np_val1


# In[31]:


#Multiplication:(9x7)
new_np_val*np_val1


# In[ ]:





# In[ ]:





# In[ ]:




