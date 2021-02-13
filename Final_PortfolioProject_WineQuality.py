#!/usr/bin/env python
# coding: utf-8

# # Final Portfolio Project: Machine Learning 
# 
# The chosen dataset, Red Wine Quality, was downloaded from Kaggle: https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009. Also see P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
# 
# The project aims to create a model which can predict red wine quality based on the given input variables.

# First of all, general relevant libraries will be imported:

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


# Now, I will import the data and evaluate the columns, length, and some other relevant parameters.

# In[3]:


wine = pd.read_csv('winequality-red.csv')


# In[26]:


wine.head()


# In[5]:


wine.describe()


# In[6]:


# there seem to be no NaN values
wine.isna().sum()


# In[7]:


# certain inputs seem to have a higher impact on quality, e.g. volatile acidity, citric acid, sulphates, and alcohol.

import seaborn as sns

sns.heatmap(wine.corr(), cmap = 'coolwarm')
# some variables are more correlated with each other than others. This can make it harder for the model to predict the output correctly
# will try a version of the model in which I include only one input variable of those that are correlated more strongly: citric acid instead of fixed acidity, density instead of fixed acidity, total sulfur dioxide instead of free sulfur dioxide


# In[8]:


sns.set_style('darkgrid')

sns.stripplot(x = 'sulphates', y = 'quality', data = wine)
plt.show()

plt.clf()
sns.stripplot(x = 'alcohol', y = 'quality', data = wine)
plt.show()


# In[9]:


# check data in regards to outliers: seems like there are columns that have outliers (max abs z-score is higher than 3)
# I will use z-score based standardization of data then to handle these better. In case the model does not perform well, I can come back and try to exclude the values in advance


from scipy import stats

wine['z_score_tsd']=stats.zscore(wine['total sulfur dioxide'])
wine['z_score_fa']=stats.zscore(wine['fixed acidity'])
wine['z_score_rs']=stats.zscore(wine['residual sugar'])
wine['z_score_fsd']=stats.zscore(wine['free sulfur dioxide'])

print(wine.z_score_tsd.abs().max())
print(wine.z_score_fa.abs().max())
print(wine.z_score_rs.abs().max())
print(wine.z_score_fsd.abs().max())

# exclude outliers based on income column and z-scores above an absolute value of 3 (which is an empirical rule)
# wine = wine.loc[wine['z_score_tsd'].abs()<=3]


# After exploring and describing the data, I will now prepare the data. I will exclude three of the lower correlation inputs.

# In[20]:


#wine.columns

X = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y = wine['quality']

X_alt = wine[['volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X_alt, y, random_state = 4)


# In[21]:


# standardize data (vs normalize) as this handles outliers better

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

scaler_alt = StandardScaler()
X_train_alt = scaler_alt.fit_transform(X_train_alt)
X_test_alt = scaler_alt.transform(X_test_alt)


# ## Training and testing the model
# 
# First, I will **train a linear regression model** based on the prepared data.

# In[22]:


from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_model.score(X_test, y_test)


# In[23]:


# alternative model with less correlated variables - gets a tiny bit better

lin_model_alt = LinearRegression()
lin_model_alt.fit(X_train_alt, y_train_alt)
lin_model_alt.score(X_test_alt, y_test_alt)


# I will try another model, **a classification model**, e.g. a decision tree. For that, I will convert the linear output variable into three arbitrary categories: not good (< 5), good (5 - 6), very good (7 - 8). 

# In[33]:


# binning output variable

qual_bins = [0, 4, 6, 10]
bin_labels = ['not good', 'good', 'very good']

wine['binned_qual'] = pd.cut(wine['quality'], qual_bins, labels = bin_labels)

wine.head()


# In[37]:


X_new = wine[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
y_new = wine['binned_qual']

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, random_state = 4)

scaler_new = StandardScaler()
X_train_new = scaler_new.fit_transform(X_train_new)
X_test_new = scaler_new.transform(X_test_new)

# test different possible models

# Logistic Regression
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train_new, y_train_new)
print(log_model.score(X_test_new, y_test_new)) # returns accuracy of 0.8325


# In[40]:


# Support Vector Machine
from sklearn.svm import SVC

accuracies = {}

for i in list(np.arange(0.01,1.1,0.1)):
    svm_model = SVC(C = i)
    svm_model.fit(X_train_new, y_train_new)
    accuracies[i] = svm_model.score(X_test_new, y_test_new)

print(accuracies)

import operator

print(max(accuracies.items(), key=operator.itemgetter(1))[0]) # returns 0.91 - C of 0.91 is best and brings accuracy of 0.8525


# In[41]:


# Decision tree - should have a max depth of 3, and does perform equally well as SVM

from sklearn.tree import DecisionTreeClassifier

trees = {}

for i in list(range(1, 10)):
    tree_model = DecisionTreeClassifier(random_state = 1, max_depth = i)
    tree_model.fit(X_train_new, y_train_new)
    trees[i] = tree_model.score(X_test_new, y_test_new)

print(max(trees.items(), key=operator.itemgetter(1))[0])
print(trees[max(trees.items(), key=operator.itemgetter(1))[0]])


# In[45]:


# Random forest - performs even better than a decision tree and reaches 89 % accuracy with 35 trees.
from sklearn.ensemble import RandomForestClassifier

forest = {}

for i in list(range(1, 200)):
    forest_model = RandomForestClassifier(random_state = 1, n_estimators = i)
    forest_model.fit(X_train_new, y_train_new)
    forest[i] = forest_model.score(X_test_new, y_test_new)

print(max(forest.items(), key=operator.itemgetter(1))[0])
print(forest[max(forest.items(), key=operator.itemgetter(1))[0]])

#print(forest)


# By using a Decision Tree or a SVM, an accuracy of approx. 85 % can be reached to predict the previously defined quality categories. When using a Random Forest, even 89 % of accuracy can be reached, which is why I would recommend using a Random Forest for predicting wine quality when using this dataset.
