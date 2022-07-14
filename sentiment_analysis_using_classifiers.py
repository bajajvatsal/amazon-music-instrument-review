# %% [markdown]
# # Amazon Music Instrument Review

# %% [markdown]
# ### About dataset
# Description of columns in the file:
# 1. reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# 2. asin - ID of the product, e.g. 0000013714
# 3. reviewerName - name of the reviewer
# 4. helpful - helpfulness rating of the review, e.g. 2/3
# 5. reviewText - text of the review
# 6. overall - rating of the product
# 7. summary - summary of the review
# 8. unixReviewTime - time of the review (unix time)
# 9. reviewTime - time of the review (raw)
# 

# %% [markdown]
# ### Importing libraries

# %%
import numpy as np
import pandas as pd
import re
import string
import torch


# %% [markdown]
# ### Import Dataset

# %%
reviews = pd.read_csv('Musical_instruments_reviews.csv')
reviews.head()

# %%
reviews.info()


# %% [markdown]
# ### Data Preprocessing

# %%
# Removing columns not required for sentiment analysis
reviews.drop(['reviewerID', 'asin', 'reviewerName', 'helpful', 'unixReviewTime', 'reviewTime'], axis=1, inplace=True)
reviews.head()


# %%
# combining summary and reviewText
reviews['text'] = reviews['reviewText'] + ' ' + reviews['summary']
reviews.drop(['reviewText', 'summary'], axis=1, inplace=True)
reviews.head()


# %%
# finding null values
reviews.isnull().sum()

# %%
reviews['text'] = reviews['text'].fillna('review_missing') # filling the missing review content with missing_review 
reviews = reviews.dropna()  # droping reviews without reviewerName
reviews.isnull().sum()

# %%
# Here assuming the 3 rating is neutral and above and below is postive and negative sentiment respectively
reviews['overall'].value_counts()

# %% [markdown]
# ### Text Preprocessing

# %%
def text_clean(txt):
    txt = str(txt).lower()
    txt = re.sub('\[.*?\]', '', txt)
    txt = re.sub('https?://\S+|www\.\S+', '', txt)
    txt = re.sub('<.*?>+', '', txt)
    txt = re.sub('[%s]' % re.escape(string.punctuation), '', txt)
    txt = re.sub('\n', '', txt)
    txt = re.sub('\w*\d\w*', '', txt)
    return txt


# %%
reviews['text'] = reviews['text'].apply(lambda x: text_clean(x))
reviews.head()


# %%
# converting ratings to sentiment class
def overall_to_sentiment_class(x):
    if x == 4.0 or x == 5.0:
        sentiment = "positive"
    elif x == 3.0:
        sentiment = "neutral"
    elif x == 1.0 or x == 2.0:
        sentiment = "negative"
    return sentiment


# %%
reviews['sentiment_class'] = reviews['overall'].apply(lambda x: overall_to_sentiment_class(x))
reviews.drop(['overall'], axis=1, inplace=True)
reviews.head()


# %% [markdown]
# ### Visualizing sentiment classes

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
list_classes_value = reviews['sentiment_class'].value_counts().to_list()
classes_name = reviews['sentiment_class'].unique()
plt.bar(classes_name,list_classes_value)


# %%
reviews['sentiment_class'].value_counts()


# %%
import nltk
stopwords = nltk.corpus.stopwords.words("english")
def stopwords_removal(txt):
    txt = ' '.join([x for x in txt.split() if x.lower() not in stopwords])
    return txt

# %%
reviews['text'] = reviews['text'].apply(lambda x: stopwords_removal(x))


# %%
reviews.head()

# %%
df_model = reviews.copy()
df_model.head()
df_model.drop(['sentiment_class'], axis=1, inplace=True)


# %%
# encoding sentiment classes into numbers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
reviews['sentiment_class'] = encoder.fit_transform(reviews['sentiment_class'])

# %%
reviews['sentiment_class'].value_counts()

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(2, 2))
X = tfidf_vectorizer.fit_transform(reviews['text'])
y = reviews['sentiment_class']


# %% [markdown]
# ### SMOTE - used as the data sample is biased towards positive sentiment class

# %%
from imblearn.over_sampling import SMOTE
from collections import Counter
smote = SMOTE(random_state=42)
X_data, y_data = smote.fit_resample(X, y)

# %% [markdown]
# ### Splitting data into test and train set

# %%
#model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state=0)


# %% [markdown]
# ### Model selection using cross_val_score evaluation method

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


logreg_model = LogisticRegression(random_state=0)
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
svc_model = SVC()
cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'KNN', 3: 'SVC'}
cv_models = [logreg_model, dt_model, knn_model, svc_model]


for i, model in enumerate(cv_models):
    print("{} Test Accuracy: {}".format(cv_dict[i], cross_val_score(model, X, y, cv=10, scoring='accuracy').mean()))


# %% [markdown]
# ### Hyper parameter tuning and model training

# %%
from sklearn.model_selection import GridSearchCV

param_grid = {'C': np.logspace(-4, 4, 50),
              'penalty': ['l1', 'l2']}
clf = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5, verbose=0, n_jobs=-1)
best_model = clf.fit(X_train, y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:", best_model.score(X_test, y_test))


# %% [markdown]
# ### Predicting results on final model

# %%
final_model = LogisticRegression(C=10000.0, random_state=0)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)
print('Accuracy of classifier on test set: {:.2f}'.format(final_model.score(X_test, y_test)))


# %% [markdown]
# ### Visualizing confusion matrix

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')



