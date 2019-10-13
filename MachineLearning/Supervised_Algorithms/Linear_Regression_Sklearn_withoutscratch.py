#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import matplotlib.pyplot as plt
#defining linear_model
from sklearn.linear_model import LinearRegression 
#mse & r2 errors
from sklearn.metrics import mean_squared_error,r2_score


# In[23]:


#generate random no set
np.random.seed(0) #with help of seed it will generate numbers which will remain same every time so we use it to fix the values of x and y
x = np.random.rand(100,1) #x data
print(x)


# In[12]:


y = 2+3*x+np.random.rand(100,1)#y data


# In[13]:


#Model initialization->call the model function eg linear regression
#using sklearn library we can do linear regression
#there are 3 methods to do it -> there is no need to calculate cost function or gradient descent it is already privided in library
lr = LinearRegression()
#fit
lr.fit(x,y)
#prediction
y_predicted=lr.predict(x) #you will get the values of y hat


# In[14]:


#model eveluation
mse = mean_squared_error(y,y_predicted)
r2 = r2_score(y,y_predicted)


# In[19]:


#printing values
print("Slope:",lr.coef_)
print("Intercept:",lr.intercept_)
print("Root mean squared error:",mse)
print("R2 score:",r2)


# In[21]:


#plotting values
#for changing
plt.figure(figsize=(10,5))
plt.scatter(x,y,s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x,y_predicted,color="r")
plt.show()


# In[ ]:




