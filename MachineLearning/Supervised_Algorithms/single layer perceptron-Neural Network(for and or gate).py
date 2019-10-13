#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import theano
import matplotlib.pyplot as plt

#defining theano variables
x=theano.tensor.matrix(name='x')
wval=np.asarray([np.random.randn(),np.random.randn()])#weight of synapse
w=theano.shared(wval,name='w')


# In[2]:


type(w)


# In[3]:


print(wval)


# In[4]:


b=theano.shared(np.random.randn(),name='b')


# In[5]:


inputs=[[0,0],
       [1,0],
       [0,1],
       [1,1]]
outputs=[0,1,1,0]#->it is not meant to solve xor gate problem in single perceptron
#for or gate outputs=[0,0,0,1] ,for and gate outputs[0,0,0,1]->it will solve this.


# In[6]:


z=theano.tensor.dot(x,w)+b#dot product


# In[7]:


#Activation function will be for every node
#applying activation function
ahat = 1/(1+theano.tensor.exp(-z))
a=theano.tensor.vector('a')


# In[8]:


cost=-(a*theano.tensor.log(ahat)+(1-a)*theano.tensor.log(1-ahat)).sum()#it is defined for 1/1+eraise to -z


# In[9]:


dcostdw = theano.tensor.grad(cost,w)
dcostdb=theano.tensor.grad(cost,b)
#apply GDA to compute the updated weights
wn=w-0.005*dcostdw
bn=b-0.005*dcostdb
#training function
train=theano.function([x,a],[ahat,cost],updates=[(w,wn),(b,bn)])


# In[10]:


cost1=[]
for i in range(60000):#iterate till 60000
    pval,costval=train(inputs,outputs)
    print(costval)
    cost1.append(costval)
    
print('the final output is')
for i in range(len(inputs)):
    print("the o/p of x1=%d and x2=%d is %.2f"%(inputs[i][0],inputs[i][1],pval[i]))


# In[11]:


plt.plot(cost1,color='red')
plt.show()


# In[ ]:





# In[ ]:




