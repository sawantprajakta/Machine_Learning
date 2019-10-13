#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
plt.figure(figsize=(10,5)) #(width,height)
plt.plot([1,2,3,4,5],[4,5,1,10,15],color='r',label='line 1',linewidth=2,marker='s',markersize=12)
plt.plot([1,2,3,4,5],[5,8,4,12,17],color='k',label='line2',linewidth=2,marker='o',markersize=12)
plt.title('Information')
plt.xlim(0,8)
plt.ylim(0,20)
#plt.axis([0,8,0,20]) #([xmin,xmax,ymin,ymax])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['line1','line2'])
#or plt.legend()
plt.show()


# In[4]:


x=[4,6,9,10,13,16]
y=[2,6,8,9,15,19]

plt.figure(figsize=(10,5))
plt.plot(x, y, 'go--', linewidth=2, markersize=12)
#plt.plot(x, y, color='green', marker='o', linestyle='dashed',linewidth=2, markersize=12)

#plt.plot(x,y,'r--')
#plt.plot(x,y,color='y',marker='s',linestyle='-.',label='abc',linewidth=4,markersize=15)
plt.title("Graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend(['line1'])

plt.show()


# In[1]:


from matplotlib import style


# In[6]:


#style.use('ggplot')


# In[4]:


from matplotlib import style
from matplotlib import pyplot as plt
style.use('ggplot')
x = [1,2,9]
y = [4,5,1]
x1 = [1,6,8]
y1 = [2,6,1]
plt.plot(x,y,'g',label="line one",linewidth=5)
plt.plot(x1,y1,'r',label='line two',linewidth=5)
plt.title("info")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True,color='k')
plt.show()


# # Bar graph
# 

# In[8]:


import matplotlib.pyplot as plt
import numpy as np


# In[9]:


company = ['A','B','C','D']
revenue =[90,136,89,50]
profit  = [30,40,34,20]

xpos = np.arange(len(company))

plt.xticks(xpos,company)
plt.ylabel("revenus=e(Bln)")
plt.title("US Tech Stocks")
plt.bar(xpos-0.2,revenue,width=0.4,label="Revenue")
plt.bar(xpos+0.2,profit,width=0.4,label="Profit")
plt.legend()
plt.show()


# In[10]:


x=[1,2,3,4,5]
y=[100,199,122,133,44]
x_name=np.arange(len(x))
y1=[24,40,116,78,63]
plt.xticks(x_name,x)
width=0.25
plt.bar(x_name ,y,label='Virat kohli',width=0.25,color='r')
plt.bar(x_name + width ,y1,label='Steve Smith',color= 'b',width=0.25)
plt.title("comparison")
plt.xlabel("number of matches")
plt.ylabel("total runs")
plt.legend()
plt.show()


# In[4]:


from matplotlib import pyplot as plt
x=[1,2,3,4,5]
y=[100,199,122,133,75]
y1=[24,40,116,78,63]
width=0.25
plt.bar(x,y,label='Virat kohli',width=0.25,color='green',alpha=0.60)
plt.bar(x,y1,label='Steve Smith',color= 'r',width=0.25,alpha=0.60)
plt.title("comparison")
plt.xlabel("number of matches")
plt.ylabel("total runs")
plt.legend()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import numpy as np

subject = ['python','ML','AI','DL','BI']
marks = [90,65,89,50,70]
plt.ylim(0,100)
plt.bar(subject,marks,color='r') #width=0.50
plt.title("JONE RECORD")
plt.xlabel('SUBJECTS')
plt.ylabel('MARKS')
plt.legend(['marks'])
plt.show()


# In[ ]:





# In[3]:


from matplotlib import pyplot as plt

x = [1,2,9,5,10,42,45,46,49,47]
y = [4,5,1,10,20,96,34,75,61,28]
x_data = [23,15,49,67,85,15,68,74,12,69,45,35]
y_data = [78,15,92,84,23,62,16,97,66,55,44,33]
plt.scatter(x_data,y_data,label='scatter',color='orange',s=100,marker='o',alpha=0.20)
plt.scatter(x,y,label='scatter1',color='y',s=100,marker='x')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()


# In[2]:


from matplotlib import pyplot as plt
import numpy as np
x=np.random.randn(200)
y=2*np.random.randn(200)
colors=np.random.rand(200)
area=500*np.random.rand()
plt.xlabel("abc")
plt.ylabel("def")
plt.title("scatter plot")

plt.scatter(x,y,marker='o',label='scatter',c=colors,s=area)
plt.legend()
plt.colorbar()
plt.show()


# # Pie charts

# In[15]:


slices=[13,25,21,19]
activities = ['sleeping','eating','working','playing']
cols=['c','m','green','orange']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow=True,
        explode=(0.1,0,0,0),
        autopct='%1.1f%%')
plt.show()


# In[2]:


import matplotlib.pyplot as plt
import numpy as np

plt.ioff() #turn interactive mode off

for i in range(3):
    plt.plot(np.random.rand(10))
    plt.show()


# In[67]:


from matplotlib import pyplot as plt
import numpy as np
import math
x = np.arange(0, math.pi*2, 0.05)
y = np.sin(x)
fig = plt.figure()

#ax = fig.add_axes([1,1,1,1])
plt.plot(x,y)
#ax.set_title("sine wave")
#ax.set_xlabel('angle')
#ax.set_ylabel('sine')
plt.show()


# In[ ]:




