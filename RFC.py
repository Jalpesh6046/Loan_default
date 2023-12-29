# Loan_Default Prediction

#Import Libraries and Datasets

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, r2_score, recall_score, classification_report, confusion_matrix
import streamlit as st

df = pd.read_csv('C:\\Users\\Jalpesh Patel\\Downloads\\Loan_default.csv')

#Make Copy of the original Data 

df1 = df.copy(deep=True)
df2 = df.copy(deep=True)

df.head()

df.tail()

# Data Preprocessing

#First I check the null values 

df.isnull().sum()

#Here in this dataset have not null values.

df.info()

df.describe()

df.columns

x = df.drop(['LoanID','Default'], axis=1)
y = df['Default']

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=20, test_size=0.30, stratify=y)

train_x.head()

train_x.reset_index(inplace=True,drop=True)
test_x.reset_index(inplace=True,drop=True)

train_y.reset_index(inplace=True,drop=True)
test_y.reset_index(inplace=True,drop=True)

train_x.shape

test_x.shape

#First We need to convert categorical features to numerical data

train_cat = train_x.select_dtypes(include="object")
train_num = train_x.select_dtypes(include="number")

test_cat = test_x.select_dtypes(include="object")
test_num = test_x.select_dtypes(include="number")

train_cat.head()

train_cat['Education'] = train_cat['Education'].map({"High School":0,"Bachelor's":1,"Master's":2,"PhD":3})
train_cat['EmploymentType'] = train_cat['EmploymentType'].map({"Unemployed":0,"Self-employed":1,"Part-time":2,"Full-time":3})
train_cat['MaritalStatus'] = train_cat['MaritalStatus'].map({"Single":0,"Divorced":1,"Married":2})
train_cat['HasMortgage'] = train_cat['HasMortgage'].map({"No":0,"Yes":1})
train_cat['HasDependents'] = train_cat['HasDependents'].map({"No":0,"Yes":1})
train_cat['LoanPurpose'] = train_cat['LoanPurpose'].map({"Other":0,"Auto":1,"Business":2,"Education":3,"Home":4})
train_cat['HasCoSigner'] = train_cat['HasCoSigner'].map({"No":0,"Yes":1})

test_cat['Education'] = test_cat['Education'].map({"High School":0,"Bachelor's":1,"Master's":2,"PhD":3})
test_cat['EmploymentType'] = test_cat['EmploymentType'].map({"Unemployed":0,"Self-employed":1,"Part-time":2,"Full-time":3})
test_cat['MaritalStatus'] = test_cat['MaritalStatus'].map({"Single":0,"Divorced":1,"Married":2})
test_cat['HasMortgage'] = test_cat['HasMortgage'].map({"No":0,"Yes":1})
test_cat['HasDependents'] = test_cat['HasDependents'].map({"No":0,"Yes":1})
test_cat['LoanPurpose'] = test_cat['LoanPurpose'].map({"Other":0,"Auto":1,"Business":2,"Education":3,"Home":4})
test_cat['HasCoSigner'] = test_cat['HasCoSigner'].map({"No":0,"Yes":1})

train_cat.head()

train_x = pd.concat([train_num, train_cat], axis=1)
test_x = pd.concat([test_num, test_cat], axis=1)

train_x.shape

test_x.shape

scaler = MinMaxScaler()
scaler.fit(train_x)
train_x = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)
test_x = pd.DataFrame(scaler.transform(test_x), columns=test_x.columns)

train_x.head()

test_x.head()

rfc = RandomForestClassifier(random_state=20)
rfc.fit(train_x, train_y)
pred_rfc = rfc.predict(test_x)
print(accuracy_score(test_y, pred_rfc))
print(classification_report(test_y, pred_rfc))

import pickle
pickle_out = open("rfc.pkl","wb")
pickle.dump(rfc, pickle_out)
pickle_out.close()





























