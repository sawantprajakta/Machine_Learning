#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import theano as th
import matplotlib.pyplot as plt


# In[10]:


#1st step to fetch data
#create some sample data and length of each should be same
xdata = np.asarray([1.2,2.0,3.6,5.8,9.11,8.51,12.55,18.52,45.12,65.12])
ydata = np.asarray([1,0,0,1,1,0,1,1,1,0])
len(ydata)


# In[11]:


X = th.tensor.vector(name="x")
y = th.tensor.vector(name="y")


# In[12]:


m = th.shared(np.random.random(),name = "m")
c = th.shared(np.random.random(),name = "c")


# In[13]:


yh = np.dot(X,m) + c


# In[14]:


n = xdata.size
cost = th.tensor.sum((y-yh)**2)/(2*n)


# In[15]:


djdm = th.tensor.grad(cost,m)
djdc = th.tensor.grad(cost,c)
mnew = m - 0.0005*djdm
cnew = c - 0.0005*djdc


# In[16]:


train = th.function([X,y],cost,updates=[(m,mnew),(c,cnew)])#X input nad Y output cost will find diff and will update the values of m and c
test = th.function([X],yh)


# In[17]:


costval = []
#iteration
for i in range(40000):
    costm=train(xdata,ydata)
    costval.append(costm)
    print(costm)


# In[18]:


a=np.linspace(0,70,20)
b = test(a)
print(b)


# In[19]:


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




