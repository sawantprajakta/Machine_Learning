#!/usr/bin/env python
# coding: utf-8

# # MULTI-CLASS TEXT CLASSIFICATION

# In[2]:


import pandas as pd
df = pd.read_csv('/home/prajakta/Downloads/Consumer_Complaints.csv')
df


# # **step2->input and output extraction

# In[3]:


from io import StringIO
df.columns
col = ['Product','Consumer complaint narrative']
df = df[col]
#Non-null values extraction
df =df[pd.notnull(df['Consumer complaint narrative'])]
df.columns = ['Product','Consumer_complaint_narrative']
df
df['category_id']=df['Product'].factorize()[0]
category_id_df = df[['Product','category_id']].drop_duplicates().sort_values('category_id')
category_id_df
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','Product']].values)
id_to_category


# In[4]:


import matplotlib.pyplot as plt
fig=plt.figure(figsize=(5,4))
df.groupby('Product').Consumer_complaint_narrative.count().plot.bar(ylim=0)
plt.show()


# In[5]:


#term frequency and inverse document frequency used acc to text
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,norm='l2',encoding='latin-1',ngram_range=(1,2),stop_words='english')
tfidf


# In[6]:


features = tfidf.fit_transform(df.Consumer_complaint_narrative)


# In[7]:


df.head()


# In[12]:


labels = df.category_id
features.shape


# In[13]:


#scikit learn feature selction by CHI2 TEST
#it measures the dependence between stochastic variables,so using this function "weeds out" the features that are the most likely to be independent of class and therefore irrelevant for classification
from sklearn.feature_selection import chi2
import numpy as np
N= 2
for Product,category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features,labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' '))==1]
    bigrams = [v for v in feature_names if len(v.split(' '))==2]
    print("# '{}':".format(Product))
    print(" .Most correlated unigrams:\n.{}".format('\n'.join(unigrams[-N:])))
    print(" .Most correlated unigrams:\n.{}".format('\n'.join(bigrams[-N:])))
    


# # Naive Bayes - MultinomialNB

# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# In[15]:


x_train,x_test,y_train,y_test = train_test_split(df['Consumer_complaint_narrative'],df['Product'],random_state=0)
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
clf = MultinomialNB()
clf.fit(x_train_tfidf, y_train)


# In[17]:


print(clf.predict(count_vect.transform(["Our account XXXX was paid in full.Enclosed evidence"])))


# In[22]:


from sklearn.svm import LinearSVC
model = LinearSVC()
x_train,x_test,y_train,y_test,indices_train,indices_test = train_test_split(features,labels,df.index,test_size=0.33,random_state=0)

model.fit(x_train,y_train)
y_pred = model.predict(x_test)
y_pred
#SVC -> score
#SVC ->confusion matrix


# In[23]:


from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[27]:


import seaborn as sns
fig,ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat,annot=True, fmt='d',xticklabels=category_id_df.Product.values,yticklabels=category_id_df.Product.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[ ]:




