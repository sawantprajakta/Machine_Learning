#!/usr/bin/env python
# coding: utf-8

# In[2]:



#sklearn.ssklearn.ensemble.RandomForestClassifier(n_estimators='warn',criterion='gini',max_depth=None
#                                                n_estimators=no.of.tree samples by default=10,max_depth=depth of each tree
 #                                               criterion=gini(impurity criteria=gini,training criteria=entropy and information gain))


# In[18]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import graphviz
import sklearn as sk
from sklearn import ensemble
from collections import Counter
import os
#for each document in training set crete a frequency matrix for this words in dictionary and corresponding labels
def make_dict(root_dir):
    all_words=[]
    emails=[os.path.join(root_dir,f)
           for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:
            for line in m:
                words=line.split(' ')
                all_words+=words
    dictionary=Counter(all_words)
    list_to_remove=list(dictionary)

    #print(list_to_remove)
    
    for item in list_to_remove:
        if item.isalpha()==False:
            del dictionary[item]
        elif len(item)==1:
            del dictionary[item]
        dictionary=dictionary.most_common(3000)
        return dictionary
TRAIN_DIR="/home/dell/Desktop/train-mails"
TEST_DIR="/home/dell/Desktop/test-mails"
dictionary=make_dict(TRAIN_DIR)
print(dictionary)
        

    


# In[16]:


def extract_features(mail_dir):
    files=[os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix=np.zeros((len(files),3000))
    train_labels=np.zeros(len(files))
    count=0;
    docID=0;
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i==2:
                    words=line.split()
                    for word in words:
                        wordID=0
                        for i,d in enumerate(dictionary):
                            if d[0]==word:
                                wordID=i
                                features_matrix[docID,wordID]=words.count(word)
            train_labels[docID]=0
            filepathTokens=fil.split('/')
            lastToken=filepathTokens[len(filepathTokens)-1]
            if lastToken.startswith("spmsg"):
                train_labels[docID]=1
                count=count+1
            docID=docID+1
    return features_matrix,train_labels
TRAIN_DIR="/home/dell/Desktop/train-mails"
TEST_DIR="/home/dell/Desktop/train-mails"
extract_features(TRAIN_DIR)
features_matrix,labels=extract_features(TRAIN_DIR)
print(features_matrix,labels)


# In[19]:


from sklearn import tree
from sklearn.metrics import accuracy_score
TRAIN_DIR="/home/dell/Desktop/train-mails"
TEST_DIR="/home/dell/Desktop/train-mails"
dictionary=make_dict(TRAIN_DIR)
print("Reading and processing emails from file.")
features_matrix,labels=extract_features(TRAIN_DIR)
test_feature_matrix,test_labels=extract_features(TEST_DIR)
model=tree.DecisionTreeClassifier()
print("Training model")
#train model
model.fit(features_matrix,labels)
predicted_labels=model.predict(test_feature_matrix)
print("Finished classifying.accuracy score:")
print(accuracy_score(test_labels,predicted_labels))


# In[ ]:





# In[ ]:




