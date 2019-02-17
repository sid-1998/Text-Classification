#!/usr/bin/env python
# coding: utf-8

# In[228]:


import sklearn.datasets as skd
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import numpy as np


# In[229]:


categories = ['0', '1']
ds = skd.load_files("./dataset/", categories=categories, encoding="UTF-8")


# In[230]:


ds.keys()


# In[231]:


ds.target_names


# In[232]:


type(ds.data)


# In[233]:


ds.data[0]


# In[ ]:





# In[234]:


def process(content):
    content = content.lower()
    content = content.strip('\n')
    content = content.replace('\n','')
    content = content.replace('\xa0',' ')
    content = content.strip('(')
    content = content.replace('(',"")
    content = content.strip(')')
    content = content.replace(')',"")
    
    word_list = word_tokenize(content)
    
    sw  = set(stopwords.words('english'))
    useful_words = [w for w in word_list if w not in sw]
    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in useful_words]
    stripped = list(filter(None, stripped))
    #stripped = list(filter("\"",stripped))
    
    l = WordNetLemmatizer()
    lem_words = []
    for w in stripped:
        lem_words.append(l.lemmatize(w))
    
    return lem_words
    
    


# In[235]:


length = len(ds.data)


# In[236]:


text_list = []
for ix in range(length):
    lem_words = process(ds.data[ix])
    text_list.append(lem_words)
    


# In[237]:


text_str = ""
final_data = []
for ix in range(length):
    text_str = "".join(text_list[ix])
    final_data.append(text_str)


# In[238]:


len(final_data)


# In[239]:


from sklearn.model_selection import train_test_split


# In[240]:


X_train, X_test, Y_train, Y_test = train_test_split(final_data, data.target, test_size=0.33)


# In[241]:


X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)


# In[242]:


print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[243]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()


# In[244]:


X_train_tf = count_vect.fit_transform(X_train)


# In[245]:


X_train_tf.shape


# In[246]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()


# In[247]:


X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)


# In[248]:


X_train_tfidf.shape


# In[249]:


from sklearn.naive_bayes import MultinomialNB


# In[251]:


clf = MultinomialNB().fit(X_train_tfidf, Y_train)


# In[252]:


X_test_tf = count_vect.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)


# In[253]:


prediction = clf.predict(X_test_tfidf)


# In[255]:


from sklearn.metrics import accuracy_score, classification_report
print(accuracy_score(Y_test, prediction))


# In[256]:


print(classification_report(Y_test, prediction, target_names=ds.target_names))


# In[257]:


from sklearn.ensemble import RandomForestClassifier


# In[258]:


rf = RandomForestClassifier().fit(X_train_tfidf, Y_train)


# In[259]:


pred_rf = rf.predict(X_test_tfidf)


# In[261]:


print( accuracy_score(Y_test, pred_rf))


# In[270]:


print(classification_report(Y_test, pred_rf, target_names=ds.target_names))


# In[262]:


from sklearn.svm import LinearSVC


# In[264]:


svc = LinearSVC().fit(X_train_tfidf, Y_train)


# In[265]:


pred_svc = svc.predict(X_test_tfidf)


# In[266]:


print(accuracy_score(Y_test, pred_svc))


# In[271]:


print(classification_report(Y_test, pred_svc, target_names=ds.target_names))


# In[ ]:




