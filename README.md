# Subject and News Classification Projects

This repository contains two machine learning projects: **Subject Classification** and **News Classification**. Both projects focus on text classification, utilizing various models and preprocessing techniques to predict categorical labels based on textual data.

## Table of Contents
1. Project Overview
   - Subject Classification
   - News Classification
2. Datasets.
3. Installation and Requirements
4. Usage
   - Preprocessing
   - Modeling
   - Evaluation.
5. Results.
6. Confusion Matrix Visualization.

## Project Overview

### Subject Classification
The **Subject Classification** project aims to classify user comments into one of three science topics:
- Biology
- Chemistry
- Physics

The dataset contains user comments and their corresponding labels (subjects). A Naive Bayes classifier is applied to predict the topic based on the comment text.

### News Classification
The **News Classification** project focuses on distinguishing between two types of news:
- Politics News
- World News

We work with both fake and true news datasets, employing models such as K-Nearest Neighbors and Random Forest for classification tasks.

---

## Datasets

### Subject Classification Dataset
- `train.csv`: Contains training data with comments and their corresponding subjects.
- `test.csv`: Contains test data with comments only (no labels).

### News Classification Dataset
- `Fake.csv`: Dataset of fake news articles.
- `True.csv`: Dataset of true news articles.
  
Both datasets have columns such as `text` (the article/comment) and `subject` (the news topic or comment topic).

---

## Installation and Requirements

To run the projects, the following Python packages are required:

```bash
pip install pandas numpy scikit-learn spacy matplotlib seaborn
python -m spacy download en_core_web_sm
```

## Usage
### Preprocessing
1. Balancing the Dataset: For the Subject Classification project, we balanced the dataset by sampling equal numbers of records for Biology, Chemistry, and Physics.

```python
min_samples = 2650
```

2. Text Preprocessing: We apply lemmatization and remove stopwords, spaces, and punctuation using the SpaCy library:

```python
def preprocess(content):
    # Apply tokenization, lemmatization, and filtering of stop words
```
## Modeling
Both projects implement multiple models for text classification:

### Subject Classification
Naive Bayes Classifier with Bag of Words (BoW)
Naive Bayes Classifier with n-grams (1, 2)

### News Classification
K-Nearest Neighbors with unigram/bigram (Euclidean and Cosine similarity)
Random Forest with trigram
Multinomial Naive Bayes with n-grams

### Evaluation
Models are evaluated using the classification_report function, which provides precision, recall, F1-score, and accuracy:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Results
### Subject Classification
For the subject classification task, the Naive Bayes model with bi-grams performed the best with the following metrics:

- Precision: 78.0
- Recall: 70.0
- F1-score: 73.0

### News Classification
In the news classification task, the Random Forest model with trigrams achieved the highest performance:

- Precision: 94.0
- Recall: 93.0
- F1-score: 93.0

## Confusion Matrix Visualization
To visualize the model's performance, a confusion matrix is plotted using seaborn:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='summer', fmt='d')
plt.show()
```



