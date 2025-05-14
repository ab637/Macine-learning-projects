import streamlit as st
import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def loan_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "No loan given"
    else:
        return "Loan given"

def main():
    st.title('🏦 Loan Approval Prediction App')

    # Taking user input for each feature (encoded or numerical)
    gender = st.text_input('Gender (1 = Male, 0 = Female)')
    married = st.text_input('Married (1 = Yes, 0 = No)')
    applicant_income = st.text_input('Applicant Income')
    education = st.text_input('Education (1 = Graduate, 0 = Not Graduate)')
    loan_amount = st.text_input('Loan Amount (in thousands)')
    self_employed = st.text_input('Self Employed (1 = Yes, 0 = No)')
    credit_history = st.text_input('Credit History (1 = Good, 0 = Bad)')
    loan_term = st.text_input('Loan Amount Term (in days, e.g., 360)')
    property_area = st.text_input('Property Area (0 = Rural, 1 = Semiurban, 2 = Urban)')
    dependents = st.text_input('Number of Dependents (0, 1, 2, 3)')
    coapplicant_income = st.text_input('Coapplicant Income')

    # Prediction button
    if st.button('Predict'):
        try:
            input_data = (
                int(gender), int(married), float(applicant_income),
                int(education), float(loan_amount), int(self_employed),
                float(credit_history), float(loan_term),
                int(property_area), int(dependents), float(coapplicant_income)
            )
            result = loan_prediction(input_data)
            st.success(f'✅ Result: {result}')
        except:
            st.error("Please enter valid input in all fields.")

# Call the main function
if __name__ == '__main__':
    main()
