import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
import pickle

st.title('Streamlit Example')

st.write("""
# Explore different classifier and datasets
Which one is the best?
""")

st.write("Titanic Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Logistic Regression')
)

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")

tot_miss=df_train.isnull().sum().sort_values(ascending=False) 
tot_cnt=df_train.isnull().count()
per_null=(tot_miss/tot_cnt)*100
per_null_precise=round(per_null,1).sort_values(ascending=False)
#print(per_null_precise)

df_null=pd.concat([tot_miss,per_null_precise],axis=1,keys=["Total_Null_Values","Null_Values_%"])
df_train.drop(['Cabin'], axis=1, inplace=True)
df_test.drop(['Cabin'], axis=1, inplace=True)

df_train.drop(['Name'], axis=1, inplace=True)
df_test.drop(['Name'], axis=1, inplace=True)

df_train.drop(['Ticket'], axis=1, inplace=True)
df_test.drop(['Ticket'], axis=1, inplace=True)

a = df_train.isnull().sum()
#print(a)

dic_sex={"male":0,"female":1}
df_train["Sex"]=df_train["Sex"].map(dic_sex)
df_test["Sex"]=df_test["Sex"].map(dic_sex)

#print(df_test)

Pclass1 = df_train[df_train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = df_train[df_train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = df_train[df_train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']

sum = int(len(df_train) + len(df_test))
#print(sum)
train_test_data = [df_train, df_test]

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

df_train["Fare"].fillna(df_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
df_test["Fare"].fillna(df_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)
#print(df_train.head(5))

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
#print(df_train.head(5))

df_train = df_train.drop(['PassengerId'], axis=1)
df_train["Age"]=df_train["Age"].replace(np.NaN,df_train["Age"].mean())
df_test["Age"]=df_test["Age"].replace(np.NaN,df_test["Age"].mean())
train_data = df_train.drop('Survived', axis=1)
target = df_train['Survived']
#print(train_data)
#print(target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np
#print(df_train)

df_train.isna().sum()
X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test.drop("PassengerId", axis=1).copy()
X_test.isna().sum()

#if classifier_name == 'SVM':
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc_svc)
filename = 'svc_model.sav'
pickle.dump(svc, open(filename, 'wb'))  

#elif classifier_name == 'Logistic Regression':
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc_log)
filename = 'lr_model.sav'
pickle.dump(logreg, open(filename, 'wb')) 

coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
    #print(coeff_df.sort_values(by='Correlation', ascending=False))

#elif classifier_name == 'KNN':
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc_knn)
filename = 'knn_model.sav'
pickle.dump(knn, open(filename, 'wb')) 






