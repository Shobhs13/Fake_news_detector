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

