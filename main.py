import numpy as np
import pickle 
#loading the model
loaded_model = pickle.load(open('trained_model.sav','rb'))


import numpy as np
import pickle

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Sample input data - Replace with actual feature values in correct order
# For example: (Gender_Male, Married_Yes, ApplicantIncome, Education_NotGraduate, LoanAmount, ...)
input_data = (1, 0, 6000, 1, 128, 0, 0, 360, 1, 0, 1)  # Example: 11 features

# Convert to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the data for prediction (1 sample, n features)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Predict the result
prediction = loaded_model.predict(input_data_reshaped)

# Print result
print("Prediction:", prediction)
if prediction[0] == 0:
    print("No loan given")
else:
    print("Loan given")
