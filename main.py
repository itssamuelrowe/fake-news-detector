#!/usr/bin/env python
# coding: utf-8

# In[96]:


# Include Libraries
# make necessary imports

import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import KFold
import itertools
import numpy as np
import seaborn as sb
import pickle


# In[98]:


# Importing dataset using pandas dataframe
# read the data
# reading data files 

fake = pd.read_csv("./fake-5000.csv")
fake.head()


# In[99]:


print(fake.shape)


# In[100]:


true = pd.read_csv("./true-5000.csv")
true.head()
print(true.shape)

# data observation
# Get shape and head
# Inspect shape of 'df' 


# In[101]:


#Data cleaning and preparation
# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'


# In[102]:


# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape


# In[103]:


# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)




# In[104]:


# Check the data
data.head()


# In[105]:


# Removing the date (we won't use it for the analysis)
data.drop(["date"],axis=1,inplace=True)
data.head()


# In[106]:


# Removing the title (we will only use the text)
#data.drop(["subject"],axis=1,inplace=True)
#data.head()


# In[107]:


# Convert to lowercase

data['text'] = data['text'].apply(lambda x: x.lower())
data.head()

# Convert to lowercase

data['title'] = data['title'].apply(lambda x: x.lower())
data.head()


# In[108]:


# Remove punctuation

import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['text'] = data['text'].apply(punctuation_removal)


# In[109]:


# Remove punctuation

import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str

data['title'] = data['title'].apply(punctuation_removal)


# In[110]:


# Check
data.head()


# In[111]:


# Print first 10 lines of 'df'
data.head(10)


# In[112]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[113]:


# Removing stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

data['title'] = data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


# In[114]:


data.head()


# In[115]:


# distribution of classes for prediction

def create_distribution(dataFile):
    return sb.countplot(x='target', data=dataFile, palette='hls')

# by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(data)


# In[116]:


# data integrity check (missing label values)
# the dataset does not contains missing values therefore no cleaning required

def data_qualityCheck():
    print("Checking data qualitites...")
    data.isnull().sum()
    data.info()  
    print("check finished.")
data_qualityCheck()


# In[117]:



# Separate the target and set up training and test datasets

# Get the target
y = data.target
y.head()


# In[118]:


# Drop the 'target' column

data.drop("target", axis=1)


# In[119]:


# Make training and test sets

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.33, random_state=53)


# In[120]:


X_train.head(10)


# In[ ]:





# In[121]:


X_test.head(10)


# In[122]:


# before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
# of unstructured data into some uniform set of attributes that an algorithm can understand. For fake news detection, it could 
# be word counts (bag of words). 

# we will start with simple bag of words technique 
# Building the Count and Tfidf Vectors

# creating feature vector - document term matrix
# Initialize the 'count_vectorizer'

count_vectorizer = CountVectorizer(stop_words='english')


# In[123]:


# Fit and transform the training data 
# Learn the vocabulary dictionary and return term-document matrix

count_train = count_vectorizer.fit_transform(X_train)


# In[124]:


print(count_vectorizer)


# In[125]:


print(count_train)


# In[126]:


# print training doc term matrix
# we have matrix of size of (4244, 56922) by calling below

def get_countVectorizer_stats():
    
    #vocab size
    print(count_train.shape)

    #check vocabulary using below command
    print(count_vectorizer.vocabulary_)

get_countVectorizer_stats()


# In[127]:


# Transform the test set

count_test = count_vectorizer.transform(X_test)


# In[128]:


# create tf-df frequency features
# tf-idf 
# Initialize a TfidfVectorizer
# Initialize the 'tfidf_vectorizer'
# This removes words which appear in more than 70% of the articles

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[129]:


# Fit and transform train set, transform test set

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)


# In[130]:


#def get_tfidf_stats():
    #tfidf_train.shape
    #get train data feature names 
    #print(tfidf_train.A[:10])

#get_tfidf_stats()


# In[131]:


# Transform the test set 

tfidf_test = tfidf_vectorizer.transform(X_test)


# In[132]:


# get feature names

# Get the feature names of 'tfidf_vectorizer'

print(tfidf_vectorizer.get_feature_names()[-10:])


# In[133]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[135]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)


# In[ ]:





# In[ ]:





# In[ ]:




