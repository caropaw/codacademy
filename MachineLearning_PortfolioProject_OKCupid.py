#!/usr/bin/env python
# coding: utf-8

# # OKCupid Portfolio Project

# First, I will read in the data and scan it in order to decide on a problem which can be solved by machine learning.

# In[1]:


# import relevant libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('profiles.csv')

data.head()


# In[3]:


#entries in dataset
print(len(data))

#description of numerical values
data.describe()


# In[4]:


#unique values in categorical columns
print(data.columns)
categ_col = ['body_type', 'diet', 'drinks', 'drugs', 'education', 'job',
       'offspring', 'orientation', 'pets', 'religion', 'sex', 'smokes', 'status']

#for i in categ_col:
#    print(data[i].unique())

print(data['education'].unique())
print(data['job'].unique())


# Based on the available data, I will try to predict the income bracket of someone based on his or her age, education, and job. I aim to use a KNN-Regressor model for this.
# 
# I will briefly check how the relevant variables of the models are distributed.

# In[5]:


plt.hist(data['age'])
plt.show()
plt.clf()

df_edu = data['education'].value_counts()
#print(df_edu)

df_job = data['job'].value_counts()
#print(df_job)

plt.hist(data['income'])
plt.show()
plt.clf()


# The last histogram showed that there seem to be outliers in the dataset, which we should exclude. In addition, at the beginning we saw that there were negative values in the column. We should check that again.

# In[6]:


# check NaN values in the income column
print(data['income'].isna().sum())

# instead of NaN values there seem to be negative values, which have to be excluded 
data = data.loc[data['income'] >= 0]

# exclude outliers based on income column and z-scores above an absolute value of 3 (which is an empirical rule)

from scipy import stats

data['z_score_income']=stats.zscore(data['income'])
data = data.loc[data['z_score_income'].abs()<=3]


# In[7]:


# check the data - seems pretty okay!
print(len(data))
print(data['income'].head())

plt.hist(data['income'])
plt.show()
plt.clf()

print(data['income'].value_counts())


# Unfortunately, it seems like there is no data from anyone with an income of 90,000. Let's see how well that plays out in the end.
# 
# Now, I will have to prepare the data:
# * save relevant columns in two new datasets: one with the input variables, and one with the target variable
# * convert categorical columns into binary values
# * create test and training sets
# * standardize data so that all variables will be equally important in the model

# In[10]:


# save relevant data
X = data[['age', 'education', 'job']]
y = data['income']

# convert categorical columns into binary values
X = pd.get_dummies(X, columns=['education'])
X = pd.get_dummies(X, columns=['job'])

# split into training and test sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 34)

# standardize data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Now that the data is ready, we can try out different classifiers to see which one works best. I will try both a KNN-Classifier as well as a Support Vector Machine (SVM).

# ### KNN-Classifier
# 
# As the data has been prepared, all that remains is training and testing the model.

# In[21]:


from sklearn.neighbors import KNeighborsClassifier

#accuracies = {}

#for i in range(1, 100):
#    model_1 = KNeighborsClassifier(n_neighbors = i)
#    model_1.fit(X_train, y_train)
#    accuracies[i] = model_1.score(X_test, y_test)

#print(accuracies)

import operator

# returns 54
#max(accuracies.items(), key=operator.itemgetter(1))[0]

model_1 = KNeighborsClassifier(n_neighbors = 54)
model_1.fit(X_train, y_train)
y_predict = model_1.predict(X_test)


# In[22]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)
print(cm)


# In[27]:


sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%')
plt.show()


# ## SVM
# 
# As the KNN Classifier was not very accurate in predicting the income, I will try out another model.

# In[33]:


from sklearn.svm import SVC

accuracies = {}

for i in list(np.arange(0.01,1.1,0.1)):
    model_2 = SVC(C = i)
    model_2.fit(X_train, y_train)
    accuracies[i] = model_2.score(X_test, y_test)

print(accuracies)


# In[34]:


max(accuracies.items(), key=operator.itemgetter(1))[0]


# While the SVM performs slightly better than the first model, it still does not perform very well. It seems like this problem could not be solved with the available data/another problem should be chosen. 
