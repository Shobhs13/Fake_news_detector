# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 21:44:39 2020

@author: HP
"""
#importing numpy 
import numpy as np
#importing pandas 
import pandas as pd
#importing intertools
import itertools
#impoting sklearn model import train_test_split function
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score , confusion_matrix

#Impoting the data
df = pd.read_csv('C:\\Users\\HP\\Downloads\\news.csv')
print(df.shape)
print(df.head())


#Defining Labels
labels=df.label
labels.head()

#Splitting of Data into test and train case.
x_train,x_test,y_train,y_test=train_test_split(df['text'],labels,test_size=0.2, random_state=7)

#Initialzing TfidfVectorizer
tfidf_vectorizer= TfidfVectorizer(stop_words='english', max_df=0.7)

#tranforing data set into test and train sets
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initializing Passive Aggresive Classifier
pac=PassiveAggressiveClassifier(max_iter=1000)
pac.fit(tfidf_train,y_train)

#predicting Data Set
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
