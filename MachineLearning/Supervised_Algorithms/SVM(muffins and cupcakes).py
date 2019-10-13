#!/usr/bin/env python
# coding: utf-8

# **Classifying Muffins and Cupcakes with SVM**

# __Step 1:__ Import Packages

# In[1]:


# Packages for analysis
import pandas as pd
import numpy as np
from sklearn import svm

# Packages for visuals
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(font_scale=1.2)

# Allows charts to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Pickle package
import pickle


# __Step 2:__ Import Data

# In[2]:


# Read in muffin and cupcake ingredient data
recipes = pd.read_csv('recipes_muffins_cupcakes.csv')
recipes


# __Step 3:__ Prepare the Data

# In[3]:


# Plot two ingredients
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70});


# In[4]:


# Specify inputs for the model
# ingredients = recipes[['Flour', 'Milk', 'Sugar', 'Butter', 'Egg', 'Baking Powder', 'Vanilla', 'Salt']].as_matrix()
ingredients = recipes[['Flour','Sugar']].as_matrix()
type_label = np.where(recipes['Type']=='Muffin', 0, 1)

# Feature names
recipe_features = recipes.columns.values[1:].tolist()
print(recipe_features)


# __Step 4:__ Fit the Model

# In[5]:


# Fit the SVM model
#model = svm.SVC(kernel='linear')
model = svm.SVC(kernel='linear', C=2**5)
#model = svm.SVC(kernel='linear', decision_function_shape='ovr')
#model = svm.SVC(kernel='rbf', C=1, gamma=2**-5)
model.fit(ingredients, type_label)


# __Step 5:__ Visualize Results

# In[6]:


# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(30, 60)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plot the parallels to the separating hyperplane that pass through the support vectors
b = model.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = model.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])


# In[7]:


# Plot the hyperplane
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black');


# In[8]:


# Look at the margins and support vectors
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=80, facecolors='none');


# __Step 6:__ Predict New Case

# In[9]:


# Create a function to guess when a recipe is a muffin or a cupcake
#var = model.predict([[flour,sugar]])
def muffin_or_cupcake(flour, sugar):
    if(model.predict([[flour, sugar]]))==0:
        print('You\'re looking at a muffin recipe!')
    else:
        print('You\'re looking at a cupcake recipe!')


# In[10]:


# Predict if 50 parts flour and 20 parts sugar
muffin_or_cupcake(50, 20)


# In[11]:


# Plot the point to visually see where the point lies
sns.lmplot('Flour', 'Sugar', data=recipes, hue='Type', palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.plot(xx, yy, linewidth=2, color='black')
plt.plot(50, 20, 'yo', markersize='9');


# In[12]:


# Predict if 40 parts flour and 20 parts sugar
muffin_or_cupcake(40,20)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




