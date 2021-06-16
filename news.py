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
from flask import Flask

# In[9]:


true = pd.read_csv(r'./true-5000.csv')
true.head()


# In[10]:


true.shape


# In[11]:


true = true[:1000]


# In[12]:


true.shape


# In[13]:


fake = pd.read_csv (r'./fake-5000.csv')
fake.head()


# In[14]:


fake.shape


# In[15]:


fake = fake[:1000]


# In[16]:


fake.shape


# In[17]:


#Data cleaning and preparation


# In[18]:


# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'


# In[19]:


# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape


# In[20]:


# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)


# In[21]:


# Check the data
data.head()


# In[22]:


#Searching for null values.

data.isna().sum()


# In[23]:


# distribution of classes for prediction

def create_distribution(dataFile):
    return sb.countplot(x='target', data=dataFile, palette='hls')

# by calling below we can see that training, test and valid data seems to be failry evenly distributed between the classes
create_distribution(data)


# In[24]:


# data integrity check (missing label values)
# the dataset does not contains missing values therefore no cleaning required

def data_qualityCheck():
    print("Checking data qualitites...")
    data.isnull().sum()
    data.info()  
    print("check finished.")
data_qualityCheck()


# In[25]:


# Separate the target and set up training and test datasets

# Get the labels
y = data.target
y.head()


# In[26]:


# Drop the 'target' column

data.drop("target", axis=1,inplace=True)


# In[27]:


# Removing the date (we won't use it for the analysis)
data.drop(["date"],axis=1,inplace=True)


# In[28]:


# Removing the title (we will only use the text)
data.drop(["title"],axis=1,inplace=True)


# In[29]:


data.head()


# In[46]:


# Make training and test sets

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], y, test_size=0.33, random_state=53)


# In[47]:


X_train.head()


# In[48]:


X_test.head()


# In[49]:


# before we can train an algorithm to classify fake news labels, we need to extract features from it. It means reducing the mass
# of unstructured data into some uniform set of attributes that an algorithm can understand. For fake news detection, it could 
# be word counts (bag of words). 

# we will start with simple bag of words technique 
# Building the Count and Tfidf Vectors

# creating feature vector - document term matrix
# Initialize the 'count_vectorizer'

count_vectorizer = CountVectorizer(stop_words='english')


# In[50]:


# Fit and transform the training data 
# Learn the vocabulary dictionary and return term-document matrix

count_train = count_vectorizer.fit_transform(X_train)


# In[51]:


print(count_vectorizer)


# In[52]:


print(count_train)


# In[53]:


# print training doc term matrix
# we have matrix of size of (1340, 22630) by calling below

def get_countVectorizer_stats():
    
    #vocab size
    print(count_train.shape)

    #check vocabulary using below command
    print(count_vectorizer.vocabulary_)

get_countVectorizer_stats()


# In[54]:


# Transform the test set

count_test = count_vectorizer.transform(X_test)


# In[55]:


# create tf-df frequency features
# tf-idf 
# Initialize a TfidfVectorizer
# Initialize the 'tfidf_vectorizer'
# This removes words which appear in more than 70% of the articles

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[56]:


# Fit and transform train set, transform test set

# Fit and transform the training data 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)


# In[59]:


def get_tfidf_stats():
    tfidf_train.shape
    #get train data feature names 
    print(tfidf_train.A[:10])

get_tfidf_stats()


# In[60]:


# Transform the test set 

tfidf_test = tfidf_vectorizer.transform(X_test)


# In[61]:


# get feature names

# Get the feature names of 'tfidf_vectorizer'

print(tfidf_vectorizer.get_feature_names()[-10:])


# In[62]:


# Get the feature names of 'count_vectorizer'

print(count_vectorizer.get_feature_names()[:10])


# In[64]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)


# In[65]:


# Check whether the DataFrames are equal

print(count_df.equals(tfidf_df))


# In[66]:


print(count_df.head())


# In[67]:


print(tfidf_df.head())


# In[68]:


# Function to plot the confusion matrix 
# This function prints and plots the confusion matrix
# Normalization can be applied by setting 'normalize=True'

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[69]:


# building classifier using naive bayes 
# Naive Bayes classifier for Multinomial model

nb_pipeline = Pipeline([
        ('NBTV',tfidf_vectorizer),
        ('nb_clf',MultinomialNB())])


# In[70]:


# Fit Naive Bayes classifier according to X, y

nb_pipeline.fit(X_train,y_train)


# In[71]:


# Perform classification on an array of test vectors X

predicted_nbt = nb_pipeline.predict(X_test)


# In[72]:


score = metrics.accuracy_score(y_test, predicted_nbt)
print(f'Accuracy: {round(score*100,2)}%')


# In[76]:


cm = metrics.confusion_matrix(y_test, predicted_nbt, labels=['fake', 'true'])
plot_confusion_matrix(cm, classes=['fake', 'true'])


# In[77]:


print(cm)


# In[78]:


nbc_pipeline = Pipeline([
        ('NBCV',count_vectorizer),
        ('nb_clf',MultinomialNB())])
nbc_pipeline.fit(X_train,y_train)


# In[79]:


predicted_nbc = nbc_pipeline.predict(X_test)
score = metrics.accuracy_score(y_test, predicted_nbc)
print(f'Accuracy: {round(score*100,2)}%')


# In[81]:


cm1 = metrics.confusion_matrix(y_test, predicted_nbc, labels=['fake', 'true'])
plot_confusion_matrix(cm1, classes=['fake', 'true'])


# In[82]:


print(cm1)


# In[83]:


print(metrics.classification_report(y_test, predicted_nbt))


# In[84]:


print(metrics.classification_report(y_test, predicted_nbc))


# In[85]:


# building Passive Aggressive Classifier 
# Applying Passive Aggressive Classifier

# Initialize a PassiveAggressiveClassifier
linear_clf = Pipeline([
        ('linear',tfidf_vectorizer),
        ('pa_clf',PassiveAggressiveClassifier(max_iter=50))])
linear_clf.fit(X_train,y_train)


# In[86]:


#Predict on the test set and calculate accuracy

pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print(f'Accuracy: {round(score*100,2)}%')

# In[87]:


#Build confusion matrix

cm = metrics.confusion_matrix(y_test, pred, labels=['fake', 'true'])
plot_confusion_matrix(cm, classes=['fake', 'true'])

print(cm)
print(metrics.classification_report(y_test, pred))


# saving best model to the disk

model_file = 'final_model.sav'
pickle.dump(linear_clf,open(model_file,'wb'))

stats = {
    accuracy: round(score * 100, 2),
    confusion_matrix: cm
}

# Serializing json 
json_object = json.dumps(stats, indent = 4)
  
# Writing to sample.json
with open("stats.json", "w+") as outfile:
    outfile.write(json_object)