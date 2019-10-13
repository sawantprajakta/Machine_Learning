#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd


x = 10 ** np.arange(1, 10)
y = x * 2
data = pd.DataFrame(data={'x': x, 'y': y})


lm = sns.lmplot('x', 'y', data, size=7, truncate=True, scatter_kws={"s": 100})


axes = lm.ax


axes.set_ylim(-1000000000,)

plt.show()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import pandas as pd


x = 10 ** np.arange(1, 10)
y = x * 2
data = pd.DataFrame(data={'x': x, 'y': y})


fig, ax = plt.subplots()


ax.set(xscale="log", yscale="log")


sns.regplot("x", "y", data, ax=ax, scatter_kws={"s": 100})

plt.show()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt


iris = sns.load_dataset("iris")


sns.swarmplot(x="species", y="petal_length", data=iris)


plt.show()


# In[ ]:




