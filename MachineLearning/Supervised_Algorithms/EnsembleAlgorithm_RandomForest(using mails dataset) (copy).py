#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import sklearn as sk
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import accuracy_score
import graphviz
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from collections import Counter
import os #to read files we use library os
def make_dict(root_dir):
    all_words = [] #list of words
    emails = [os.path.join(root_dir,f)#read the file
             for f in os.listdir(root_dir)]
    for mail in emails:
        with open(mail) as m:#m is alias for open(mail)
            for line in m:#from mail it will go to each line
                words = line.split(' ')#split the words from lines
                all_words += words #wordsl will concatinate
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:#it will remove numerical value from every line-> alpha is a numerical value
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
        dictionary = dictionary.most_common(3000)#dict carrying most top 3000 words
        return dictionary

TRAIN_DIR = "/home/prajakta/Desktop/train-mails"
TEST_DIR = "/home/prajakta/Desktop/test-mails"
dictionary = make_dict(TRAIN_DIR)
print(dictionary)


# In[3]:


from sklearn.metrics import accuracy_score
def extract_features(mail_dir):#we will make words to features
    files = [os.path.join(mail_dir,fi)for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    train_labels = np.zeros(len(files))
    count = 0;
    docID = 0;
    for fil in files:
        with open(fil) as fi:
            for i,line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i,d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID,wordID] = words.count(word)
            train_labels[docID] = 0
            filepathTokens = fil.split('/')
            lastToken = filepathTokens[len(filepathTokens) - 1]
            if lastToken.startswith("spmsg"):
                train_labels[docID] = 1
                count = count + 1
            docID = docID +1
    return features_matrix,train_labels
TRAIN_DIR = "/home/prajakta/Desktop/train-mails"
TEST_DIR = "/home/prajakta/Desktop/test-mails"
features_matrix,labels = extract_features(TRAIN_DIR)
print(features_matrix,labels)


# In[5]:


from sklearn.ensemble import RandomForestClassifier
TRAIN_DIR = "/home/prajakta/Desktop/train-mails"
TEST_DIR = "/home/prajakta/Desktop/test-mails"
dictionary = make_dict(TRAIN_DIR)
print("reading and processing emails from file")
#creating train data set
features_matrix, labels = extract_features(TRAIN_DIR)
#creating test data set
test_feature_matrix, test_labels = extract_features(TEST_DIR)
#Random forest classifier
model = RandomForestClassifier()
print("Training model")
#train model
model.fit(features_matrix, labels)#spam and not spam labels
predicted_labels = model.predict(test_feature_matrix)
print("FINISHED classifying.accuracy score:")
print(accuracy_score(test_labels, predicted_labels))


# In[ ]:





# In[ ]:





# In[ ]:




