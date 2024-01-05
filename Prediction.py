import pickle
import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, accuracy_score
from streamlit_option_menu import option_menu


pickle_in = open("rfc.pkl", "rb")
rfc = pickle.load(pickle_in)
 
def map_categorical_inputs(Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner):
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    employment_mapping = {"Unemployed": 0, "Self-employed": 1, "Part-time": 2, "Full-time": 3}
    marital_mapping = {"Single": 0, "Divorced": 1, "Married": 2}
    mortgage_mapping = {"No": 0, "Yes": 1}
    dependents_mapping = {"No": 0, "Yes": 1}
    purpose_mapping = {"Other": 0, "Auto": 1, "Business": 2, "Education": 3, "Home": 4}
    co_signer_mapping = {"No": 0, "Yes": 1}
 
    mapped_education = education_mapping.get(Education, -1)
    mapped_employment = employment_mapping.get(EmploymentType, -1)
    mapped_marital = marital_mapping.get(MaritalStatus, -1)
    mapped_mortgage = mortgage_mapping.get(HasMortgage, -1)
    mapped_dependents = dependents_mapping.get(HasDependents, -1)
    mapped_purpose = purpose_mapping.get(LoanPurpose, -1)
    mapped_co_signer = co_signer_mapping.get(HasCoSigner, -1)
 
    return mapped_education, mapped_employment, mapped_marital, mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer
 
def predict_output(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate, LoanTerm, DTIRatio,
                   mapped_education, mapped_employment, mapped_marital, mapped_mortgage, mapped_dependents,
                   mapped_purpose, mapped_co_signer):
    prediction = rfc.predict([[Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate,
                               LoanTerm, DTIRatio, mapped_education, mapped_employment, mapped_marital,
                               mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer]])
    prediction_label = "Defaulter" if prediction[0] == 1 else "Not Defaulter"
    
    return prediction_label
 
def main():
    html_temp = """
<div style ="background-color: skyblue; padding: 1px">
<h2 style ="color: black; text-align:center;">Loan Defaulter Prediction </h2>
</div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.write("&nbsp;")
    col1, col2, col3 = st.columns(3)
    with col1:
        Name= st.text_input("Name of Applicant *")
        Age= st.text_input("Age (Years) *")
        MaritalStatus=st.selectbox("Marital Status *", ["Single", "Divorced", "Married"])
        Education=st.selectbox("Education *", ["High School", "Bachelor's", "Master's", "PhD"])
        EmploymentType=st.selectbox("Employment Type *", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
        MonthsEmployed=st.text_input("Months Employed *")    
    with col2:
        LoanAmount=st.text_input("Loan Amount (USD) *")
        LoanTerm=st.text_input("Loan Term (Months) *")
        LoanPurpose=st.selectbox("Loan Purpose *", ["Other", "Auto", "Business", "Education", "Home"])
        InterestRate=st.text_input("Interest Rate (%) *")
        Income= st.text_input("Annual Income (USD) *")
        HasDependents=st.selectbox("Has Dependents? *", ["No", "Yes"])
    with col3:
        HasMortgage=st.selectbox("Has Mortgage? *", ["No", "Yes"])
        NumCreditLines=st.text_input("Number of Credit Lines *")
        CreditScore=st.text_input("Credit Score *")
        DTIRatio=st.text_input("DTI Ratio *")
        HasCoSigner=st.selectbox("Has CoSigner? *", ["No", "Yes"])
    mapped_education, mapped_employment, mapped_marital, mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer = map_categorical_inputs(
        Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner)
    result = ""
    col4,col5,col6 = st.columns([5,6,6])
    with col5:
        if st.button("Predict"):
            result = predict_output(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate,
                                LoanTerm, DTIRatio, mapped_education, mapped_employment, mapped_marital,
                                mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer)
            st.success(f"The applicant is {result}")
            with st.sidebar:
                df = pd.read_csv("C:\\Users\\Jalpesh Patel\\Downloads\\Loan_default.csv")
             
                df = df.dropna()
             
                df1 = df.copy()
             
                df['Education'] = df['Education'].map({"High School":0,"Bachelor's":1,"Master's":2,"PhD":3})
                df['EmploymentType'] = df['EmploymentType'].map({"Unemployed":0,"Self-employed":1,"Part-time":2,"Full-time":3})
                df['MaritalStatus'] = df['MaritalStatus'].map({"Single":0,"Divorced":1,"Married":2})
                df['HasMortgage'] = df['HasMortgage'].map({"No":0,"Yes":1})
                df['HasDependents'] = df['HasDependents'].map({"No":0,"Yes":1})
                df['LoanPurpose'] = df['LoanPurpose'].map({"Other":0,"Auto":1,"Business":2,"Education":3,"Home":4})
                df['HasCoSigner'] = df['HasCoSigner'].map({"No":0,"Yes":1})
               
                x = df.drop(['LoanID','Default'], axis=1)
                y = df['Default']
             
             
                train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=20, test_size=0.30, stratify=y)

                pred_y = rfc.predict(test_x)
             
                score = accuracy_score(test_y, pred_y)
                
                # Display the pie chart using Plotly Express without legend and with white background for 'Incorrect'
                fig = px.pie(values=[score, 1 - score], names=['Correct', 'Incorrect'], color_discrete_sequence=['green', 'white'], labels='Probability')
                fig.update_traces(textinfo='none' )  
                #fig.update_layout(font=dict(color='#FFFFFF'), title=(f'{result} with Probability\n{accuracy_score(test_y, pred_y):.2f}', ))
                st.markdown(f"<h3 style='text-align: center;'>The probability of applicant being {result} <br>is {accuracy_score(test_y, pred_y):.2f}</h3>", unsafe_allow_html=True)
                fig.update_layout(showlegend=False)
                fig.update_layout(width=290, height=350)
                st.plotly_chart(fig)

    with col6:
        if st.button("View Dashboard"):
            switch_page("Dashboard")

 
if __name__ == "__main__":
    main()

    
    
 

