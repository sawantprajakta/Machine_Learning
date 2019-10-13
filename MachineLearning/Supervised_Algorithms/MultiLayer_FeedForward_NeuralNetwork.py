#!/usr/bin/env python
# coding: utf-8

# In[1]:


#multi layer perceptron with back propogation
import numpy as np
import theano
import matplotlib.pyplot as plt


# In[2]:


inputs=[[0,0],
       [1,0],
       [0,1],
       [1,1]]
outputs=[1,0,0,1]


# In[3]:


x=theano.tensor.matrix(name='x')


# In[4]:


#Hidden layer as inputs from every neuron are 2 and we have 3 neuron
w1val=np.asarray([np.random.randn(),np.random.randn()])#weight of synapse
w1=theano.shared(w1val,name='w1')
w2val=np.asarray([np.random.randn(),np.random.randn()])#weight of synapse
w2=theano.shared(w2val,name='w2')
w3val=np.asarray([np.random.randn(),np.random.randn()])#weight of synapse
w3=theano.shared(w3val,name='w3')


# In[5]:


#Bias value is 1
b1 = theano.shared(1.1,name='b1')
b2 = theano.shared(1.2,name='b2')
b3 = theano.shared(1.3,name='b3')


# In[6]:


#computation foe every neuron
#hidden layer
a1sum=theano.tensor.dot(x,w1)+b1
a2sum=theano.tensor.dot(x,w2)+b2

a1=1/(1+theano.tensor.exp(-1*a1sum))
a2=1/(1+theano.tensor.exp(-1*a2sum))

#output layer neuron
#stack is combining two hiding layer values & feeding to the output layer
x2 = theano.tensor.stack([a1,a2],axis=1)


# In[7]:


'''if we write
[[a11,a12,a21,a22],[a33,a34,a43,a44]]-> inputs
what stack will do is
[a11,a33],[a12,a34],[a21,a43],[a22,a44]'''

a3sum=theano.tensor.dot(x2,w3)+b3
a3=1/(1+theano.tensor.exp(-1*a3sum))

#final output
ahat=a3

#actual output
a=theano.tensor.vector(name='a')


# In[8]:


#cost function
cost=-(a*theano.tensor.log(ahat)+(1-a)*theano.tensor.log(1-ahat)).sum()#it is defined for 1/1+eraise to -z
#GDA role
#for calculating gradient

dcostdw1 = theano.tensor.grad(cost,w1)
dcostdw2 = theano.tensor.grad(cost,w2)
dcostdw3 = theano.tensor.grad(cost,w3)

dcostdb1=theano.tensor.grad(cost,b1)
dcostdb2=theano.tensor.grad(cost,b2)
dcostdb3=theano.tensor.grad(cost,b3)

#apply GDA to update the weights
wn1=w1-0.02*dcostdw1
wn2=w2-0.02*dcostdw2
wn3=w3-0.02*dcostdw3

wb1=b1-0.02*dcostdb1
wb2=b2-0.02*dcostdb2
wb3=b3-0.02*dcostdb3
#theano function for training the algorithm
train=theano.function([x,a],[ahat,cost],updates=[(w1,wn1),(w2,wn2),(w3,wn3),(b1,wb1),(b2,wb2),(b3,wb3)])

cost1=[]
val1=[]

#training a model
for i in range(25000):
    pval,costval=train(inputs,outputs)
    print(costval)
    val1.append(pval)
    cost1.append(costval)


# In[9]:


print('the final outputs are:')
for i in range(len(inputs)):
    print("the output of x1=%d | x2=%d is %.2f"%(inputs[i][0],inputs[i][1],pval[i]))
plt.plot(cost1,color='red')
plt.show()


# In[ ]:





# In[ ]:




