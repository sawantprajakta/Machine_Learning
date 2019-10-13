#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Using Theano

# In[14]:


import numpy as np
import theano as th
import matplotlib.pyplot as plt


# In[15]:


#1st step to fetch data
#create some sample data and length of each should be same
xdata = np.asarray([1.2,2.0,3.6,5.8,9.11,8.51,12.55,18.52,45.12,65.12])
ydata = np.asarray([8.9,12.56,24.21,32.12,7.56,12.65,35.65,41.1,21.4,44.88])


# In[16]:


len(ydata)


# In[17]:


len(xdata)


# In[18]:


#use x and y for pre processing
#we need to vectorize the arrays into vectors bcoz theano deals with the vectors
#to remove error u can do scaling as well
#define algorithm -> y = mX+ c
#define theano variables and it deals with the theano vectors
X = th.tensor.vector(name="x")
y = th.tensor.vector(name="y")


# In[19]:


#values of m and c needs to be defined
m = th.shared(np.random.random(),name = "m")
c = th.shared(np.random.random(),name = "c")
#or we can also define our own values instead of random no
#m = 0.9876556
#c = 12.87557887


# In[20]:


#Equation of the algorithm
#yh = mX + c #here we need to define dot product between m and X that will be scalar product
yh = np.dot(X,m) + c


# In[21]:


#now we need to feed the data for training 
#cost function -> 1/2n sum (y - yh)sqr
n = xdata.size
cost = th.tensor.sum((y-yh)**2)/(2*n) #cost function


# In[22]:


#Gradient decent algorithm to need to modify cost function
#B1->m, B0->c
#mnew = m - 0.005*th.tensor.grad(cost,m) or
djdm = th.tensor.grad(cost,m)
djdc = th.tensor.grad(cost,c)
mnew = m - 0.0005*djdm
cnew = c - 0.0005*djdc


# In[23]:


#define train and test function
train = th.function([X,y],cost,updates=[(m,mnew),(c,cnew)])#X input nad Y output cost will find diff and will update the values of m and c
test = th.function([X],yh)  # X is input and yh is the predicted output                 


# In[24]:


costval = []
#iteration
for i in range(40000):
    costm=train(xdata,ydata)
    costval.append(costm)
    print(costm)


# In[25]:


a=np.linspace(0,70,20)
b = test(a)
print(b)


# In[26]:


print("final value of m is "+str(m.get_value()))
print("final value of c is "+str(c.get_value()))
print(test([65.12]))
plt.scatter(xdata,ydata,color="b",label = "data")
plt.plot(a,b,color = 'red',label = 'regression')
plt.title("Linear Regression")
plt.xlabel("xdata")
plt.ylabel("ydata")
plt.legend(loc=4)
plt.show()


# In[ ]:





# In[ ]:




