#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
#defining linear_model
from sklearn.linear_model import LinearRegression
#mse and r2 errors
from sklearn.metrics import mean_squared_error,r2_score


# In[22]:


#generate random data
np.random.seed(0)    #it will generate same values
x=np.random.rand(100,1)
print(x)


# In[23]:


y=2+3*x+np.random.rand(100,1)


# In[7]:


#model initialization
regression_model=LinearRegression()
#Fit the data(train the model)
regression_model.fit(x,y)
#predict
y_predicted=regression_model.predict(x)


# In[10]:


#model evaluation
mse=mean_squared_error(y,y_predicted)
r2=r2_score(y,y_predicted)


# In[12]:


#printing values
print('Slope:',regression_model.coef_)
print('Intercept:',regression_model.intercept_)
print('Mean Square Error:',mse)
print('R2 Square:',r2)


# In[14]:


#plotting graph
plt.figure(figsize=(10,5))  #for changing the figure
plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')
#predicted values
plt.plot(x,y_predicted,color='k')
plt.show()


# In[ ]:




