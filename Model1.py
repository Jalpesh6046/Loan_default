import pickle
import streamlit as st

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
    st.title("Welcome To Loan Default Prediction")
    html_temp = """
    <div style ="background-color: tomato; padding: 10px">
    <h2 style ="color: white; text-align:center;">Loan Defaulter Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    Age = st.text_input("Age", "Type Here")
    Income = st.text_input("Income", "Type Here")
    LoanAmount = st.text_input("LoanAmount", "Type Here")
    CreditScore = st.text_input("CreditScore", "Type Here")
    MonthsEmployed = st.text_input("MonthsEmployed", "Type Here")
    NumCreditLines = st.text_input("NumCreditLines", "Type Here")
    InterestRate = st.text_input("InterestRate", "Type Here")
    LoanTerm = st.text_input("LoanTerm", "Type Here")
    DTIRatio = st.text_input("DTIRatio", "Type Here")
    Education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    EmploymentType = st.selectbox("EmploymentType", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
    MaritalStatus = st.selectbox("LoanTerm", ["Single", "Divorced", "Married"])
    HasMortgage = st.radio("HasMortgage", ["No", "Yes"])
    HasDependents = st.radio("HasDependents", ["No", "Yes"])
    LoanPurpose = st.selectbox("LoanPurpose", ["Other", "Auto", "Business", "Education", "Home"])
    HasCoSigner = st.radio("HasCoSigner", ["No", "Yes"])
    
    mapped_education, mapped_employment, mapped_marital, mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer = map_categorical_inputs(
        Education, EmploymentType, MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner)
    
    result = ""
    
    if st.button("Predict"):
        result = predict_output(Age, Income, LoanAmount, CreditScore, MonthsEmployed, NumCreditLines, InterestRate,
                                LoanTerm, DTIRatio, mapped_education, mapped_employment, mapped_marital,
                                mapped_mortgage, mapped_dependents, mapped_purpose, mapped_co_signer)
    st.success(f'Person of Required Loan is {result}')

if __name__ == "__main__":
    main()
