#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.datasets as skd
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
from nltk.stem import SnowballStemmer


# In[2]:


categories = ['0', '1']
ds = skd.load_files("./dataset/", categories=categories, encoding="UTF-8")


# In[3]:


ds.keys()


# In[4]:


ds.target_names


# In[5]:


type(ds.data)


# In[6]:


ds.data[0]


# In[7]:


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", "  ", text)
    text = re.sub(r"\-", "  ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text 


# In[10]:


length = len(ds.data)


# In[11]:


final_data = []
for ix in range(length):
    text_str = clean_text(ds.data[ix])
    final_data.append(text_str)


# In[12]:


len(final_data)


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(final_data, ds.target, test_size=0.33)


# In[15]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[16]:


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()


# In[18]:


X_train_tf = count_vect.fit_transform(X_train)


# In[19]:


X_train_tf.shape


# In[46]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


# In[47]:


X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)


# In[22]:


X_train_tfidf.shape


# In[23]:


from sklearn.naive_bayes import MultinomialNB


# In[24]:


clf = MultinomialNB().fit(X_train_tfidf, Y_train)


# In[25]:


X_test_tf = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)


# In[26]:


prediction = clf.predict(X_test_tfidf)


# In[27]:


from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_test, prediction))


# In[28]:


print(classification_report(Y_test, prediction, target_names=ds.target_names))


# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rf = RandomForestClassifier().fit(X_train_tfidf, Y_train)


# In[31]:


pred_rf = rf.predict(X_test_tfidf)


# In[32]:


print( accuracy_score(Y_test, pred_rf))


# In[33]:


print(classification_report(Y_test, pred_rf, target_names=ds.target_names))


# In[34]:


from sklearn.svm import LinearSVC


# In[35]:


svc = LinearSVC().fit(X_train_tfidf, Y_train)


# In[36]:


pred_svc = svc.predict(X_test_tfidf)


# In[37]:


print(accuracy_score(Y_test, pred_svc))


# In[38]:


print(classification_report(Y_test, pred_svc, target_names=ds.target_names))


# In[ ]:




