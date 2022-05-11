#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # pandas is used for Linear Algebra
import numpy as np #numpy is used for Data Processing
import matplotlib.pyplot as plt #matplotlib is used for Visualization
import seaborn as sns #seaborn for statistical visualization
from sklearn.feature_selection import RFE #feature selection
from sklearn.linear_model import LogisticRegression #LogisticRegression
from sklearn.model_selection import train_test_split #Splitingdatasetbetweentrain&test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #ReportsPurpose
from sklearn.cluster import KMeans


df = pd.read_csv('data_cleaned_2021.csv')
print (df.columns.tolist())


# print(df.columns.tolist())
# df['Avg Salary(K)'].hist(bins = 20)

# In[31]:


df.head()


# In[33]:


print("rows in the dataset:",df.shape[0])
print("columns in the dataset:",df.shape[1])
df.isnull().sum()


# In[34]:


print (df.describe())
df.info()


# Showing the job title with their average salary on the barplot

# In[4]:


salary_labels = ['Lower Salary', 'Upper Salary', 'Avg Salary(K)']
salary_by_title = df.groupby('job_title_sim')[salary_labels].mean()
print(salary_by_title)
salary_by_title.plot.bar(title= "Job Title Vs Average Salary")


# Trying to see the average salary of higher rated companies and the average salary of all companies.

# In[5]:


higherrating = df[df.Rating > 4.0]
print(higherrating[['job_title_sim', 'Salary Estimate', 'Company Name', 'Rating']])
print('Average Salary of Higher Rated Companies:', higherrating['Avg Salary(K)'].mean())
print('Average Salary of All Companies:', df['Avg Salary(K)'].mean())


# Top 5 companies that are hiring data engineer

# In[18]:


df_engineer = df[df.job_title_sim.str.contains('engineer')]


# In[19]:


df_engineer.job_title_sim.value_counts()


# In[21]:


plt.barh(df_engineer['Company Name'].value_counts().index[:5] , df_engineer['Company Name'].value_counts().values[:5])


# In this dataset I am trying to apply Recursive Feature Elimination(RFE) for feature selection. I will be trying to find the top three features for making any upcoming predictions

# In[7]:


df['Avg Salary(K)'] = df['Avg Salary(K)'].astype(int) 
knn_df = df[['Python', 'spark', 'aws', 'excel','sql','sas',
             'keras','pytorch','scikit','tensor','hadoop','tableau',
             'bi','flink','mongo','google_an','Avg Salary(K)']]

features = np.array(['Python', 'spark', 'aws', 'excel','sql','sas','keras',
                     'pytorch','scikit','tensor','hadoop','tableau','bi','flink','mongo','google_an'])

X = knn_df.loc[:, features]
y = knn_df.iloc[:, 16].values
# How good is it for predicting with Logistic Regression?:
classifier = LogisticRegression(max_iter=2000)
# standard practice: break the dataset apart into test/train:
# we will use this for doing predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                   random_state=0)

print("How good is 'LogisticRegression' with all 17 features?")
print('The features:')
print(features)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f'Accuracy = {str(accuracy_score(y_test, y_pred))}')
# Now with RFE:
for num_components in range(3, 0, -1):
   print(f"\n\nShowing RFE with only {num_components} features.")
   # create a base classifier used to evaluate a subset of attributes
   model = LogisticRegression(max_iter=700000)
   # create the RFE model and select 3 attributes
   rfe = RFE(model, n_features_to_select=num_components)
   X = df.loc[:, features]
   rfe = rfe.fit(X, y)

   print('RFE (Recursive Feature Elimination) gives the following results:')
   # summarize the selection of the attributes:
   print(rfe.support_)
   print(rfe.ranking_)
   print(f'\nIn other words, these are the top {num_components} features:')
   new_features = features[rfe.support_]
   print(new_features)

   X = df.loc[:, new_features]
   # y stays the same
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                       random_state=0)
   classifier.fit(X_train, y_train)
   y_pred = classifier.predict(X_test)

 


#  Calculate the relationship between different skill set and average salary to check how closely they are related

# In[9]:


corr=knn_df.corr()
corr.style.background_gradient(cmap='coolwarm')


# Relationship between average salary with degree, job title and skills such as Python, sql

# In[47]:


pd.pivot_table(df, index = (['job_title_sim','Degree']), columns = (['Python','sql']), values = 'Avg Salary(K)', aggfunc = ['mean','count']).rename(columns={"mean":"Avg Salary(K)"})


# Using average salary display top 5 Location having highest average salary ?

# In[16]:


highest_avgsalary = df.groupby(df['Location'])['Avg Salary(K)'].mean().sort_values(ascending = False)
top5Location = highest_avgsalary.head(5)
print(top5Location)


# Calculating average salary for different degree

# In[23]:


df.Degree.unique()


# In[27]:


req_df = df[['Degree','Avg Salary(K)']]
group_df = req_df.groupby('Degree')
print(group_df.mean())


# Trend of average salary throughout different years 

# In[40]:


df.Founded.unique()


# In[39]:


req_df = df[['Founded','Avg Salary(K)']]
neg_year = req_df[req_df['Founded'] == -1].index
req_df.drop(neg_year, inplace = True)
req_df.plot.scatter(x = 'Founded', y = 'Avg Salary(K)', c = 'red' )


# In[ ]:




