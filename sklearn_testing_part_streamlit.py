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
import pandas as pd

st.title('Streamlit Example')

st.write("""
# Explore different classifier 
""")

st.write("Titanic Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNN', 'SVM', 'Logistic Regression')
)


Age = st.number_input('Age', min_value=1, max_value=100, value=25)
Sex = st.selectbox('Sex', ['male', 'female'])
Pclass= st.number_input('P Class', 1,3)
SibSp=  st.selectbox('Number of Siblings And Spouse',[0,1,2,3,4,5,8])
Parch= st.selectbox('Parch',[0,1,2,3,4,5,6])
Fare=  st.slider('Fare', 0,600)
Embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

input_dict = {'Age' : Age, 'Sex' : Sex, 'Pclass':Pclass,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Embarked':Embarked}
input_df = pd.DataFrame([input_dict])

dic_sex={"male":0,"female":1}
input_df["Sex"]= input_df["Sex"].map(dic_sex)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
input_df['Embarked'] = input_df['Embarked'].map(embarked_mapping)

print(input_df)

st.dataframe(input_df) 

if classifier_name == 'Logistic Regression':
    model = pickle.load(open('lr_model.sav','rb'))
    output = model.predict(input_df)
    output = str(output)

    st.success('The output is {}'.format(output))

elif classifier_name == 'SVM':
    model = pickle.load(open('svc_model.sav','rb'))
    output = model.predict(input_df)
    output = str(output)

    st.success('The output is {}'.format(output))   

elif classifier_name == 'KNN':
    model = pickle.load(open('knn_model.sav','rb'))
    output = model.predict(input_df)
    output = str(output)

    st.success('The output is {}'.format(output))       
