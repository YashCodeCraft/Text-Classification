# -*- coding: utf-8 -*-
"""NLP text_classification

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m-EYH2x1llXWIVWUGSlMOTuVijPmahiX

#Subject classificattion
"""

import pandas as pd

train_data = pd.read_csv('dataset in csv format')
test_data = pd.read_csv('dataset in csv format')

print(train_data.shape)
print(test_data.shape)

8695+1586

train_data.head(5)

"""__Preprocessing__"""

# connecting training and testing data

df = pd.concat([train_data, test_data], axis=0)
df.shape

df.Topic.value_counts()

#Balancing the imbalance data

min_samples = 2650

bio_data = df.loc[df['Topic'] == 'Biology'].sample(min_samples, random_state=42)
chem_data = df.loc[df['Topic'] == 'Chemistry'].sample(min_samples, random_state=42)
phy_data = df.loc[df['Topic'] == 'Physics'].sample(min_samples, random_state=42)

bal_data = pd.concat([bio_data, chem_data, phy_data], axis=0)
bal_data.Topic.value_counts()

# encoding topic into numbers

bal_data = bal_data.drop("Id", axis=1)
bal_data['Topic_n'] = bal_data.Topic.map(
    {"Biology": 0, "Chemistry": 1, "Physics": 2}
)
print(bal_data.Topic.value_counts())
print()
print(bal_data.Topic_n.value_counts())

"""__ML model__"""

# train and test

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    bal_data.Comment,
    bal_data.Topic_n,
    stratify=bal_data.Topic_n, # balances the the praportion after splitting
    random_state=42,
    test_size=0.2
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# BOW

clf = Pipeline([
    ('vector', CountVectorizer()),
    ('model', MultinomialNB())
])

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))

# n grams 1, 2

clf = Pipeline([
    ('vector', CountVectorizer(ngram_range=(1, 2))),
    ('model', MultinomialNB())
])

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))

x_test

y_test

y_pred

# Now preprocess the content in dataframe

import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess(content):
  filter = []
  doc = nlp(content)

  for token in doc:
    if token.is_stop or token.is_space or token.is_punct:
      continue
    filter.append((token.lemma_).lower())
  return " ".join(filter)



bal_data

bal_data['processed_data'] = bal_data.Comment.apply(preprocess)

bal_data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    bal_data.processed_data,
    bal_data.Topic_n,
    stratify=bal_data.Topic_n, # balances the the praportion after splitting
    random_state=42,
    test_size=0.2
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# BOW

clf = Pipeline([
    ('vector', CountVectorizer()),
    ('model', MultinomialNB())
])

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print(classification_report(y_test, y_pred))

"""__Confusion matrix__"""

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4),)
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    cmap="summer",
    annot=True,
    fmt = 'd',
    linecolor='black'
)
plt.show()

"""# news classification"""

import pandas as pd

fake = pd.read_csv("dataset in csv format")
true = pd.read_csv('dataset in csv format')

print(fake.shape)
print(true.shape)

fake.tail(5)

fake.columns

true.tail(5)

true.columns

print(true.subject.value_counts())
print()
print(fake.subject.value_counts())

# encoding

true['en_sub'] = true.subject.map({"politicsNews":0, "worldnews":1})
true.head()

true.isnull().sum()

"""__Modelling without Pre-processing Text data__"""

# Modelling without Pre-processing Text data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    true.text, true.en_sub, random_state=2022, stratify= true.en_sub, train_size=0.8
)

print(x_train.shape)
print(y_test.value_counts())
print(y_train.value_counts())

# unigram

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

kn = Pipeline([
    ('vector', CountVectorizer()),
    ('knn_mod', KNeighborsClassifier(n_neighbors=10, metric='euclidean')),
])

kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(classification_report(y_test, y_pred))

# bigram with cosine

kn = Pipeline([
    ('vector', CountVectorizer(ngram_range=(1, 2))),
    ('knn_mod', KNeighborsClassifier(n_neighbors=10, metric='cosine')),
])

kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(classification_report(y_test, y_pred))

# trigram with randoom forest

from sklearn.ensemble import RandomForestClassifier

kn = Pipeline([
    ('vector', CountVectorizer()),
    ('rf_mod', RandomForestClassifier(n_estimators=100)),
])

kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(classification_report(y_test, y_pred))

# unigram and bi using multinomial

from sklearn.naive_bayes import MultinomialNB

kn = Pipeline([
    ('vector', CountVectorizer(ngram_range=(1, 2))),
    ('naive_mod', MultinomialNB(alpha=0.75)),
])

kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(classification_report(y_test, y_pred))

"""Modelling with Pre-processing Text data"""

import spacy
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
  doc = nlp(text)
  filter = []
  for token in doc:
    if token.is_space or token.is_punct or token.is_stop:
      continue
    filter.append((token.lemma_).lower())
  return " ".join(filter)

true['pre_text'] = true.text.apply(preprocess)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    true.pre_text, true.en_sub, random_state=2022, stratify= true.en_sub, train_size=0.8
)

# trigram with random forest

from sklearn.ensemble import RandomForestClassifier

kn = Pipeline([
    ('vector', CountVectorizer(ngram_range=(3, 3))),
    ('rf_mod', RandomForestClassifier(n_estimators=50)),
])

kn.fit(x_train, y_train)
y_pred = kn.predict(x_test)
print(classification_report(y_test, y_pred))

