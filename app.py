import streamlit as st
import numpy as np
#import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
from tensorflow.keras.models import load_model


# Load the train model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "mode.h5")

model = load_model(model_path)
#model = load_model(file_path)


## Lode the encoders

with open('label_encoder.pkl','rb')as file:
    label_encode_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb')as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl','rb')as file:
    scaler = pickle.load(file)


## Streamlit app

st.title("Customer churn Prediction")

# User Input
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',label_encode_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimeted_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_product = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])


# Prepare the input data

input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encode_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_product],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimeted_salary]

})

## One_hot encoder for 'Geography'

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


## Combine one-hot encoded columns with data

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df.reset_index(drop=True)],axis=1)

## Scale the input data

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
prediction_probab = prediction[0][0]

if prediction_probab > 0.5:
    st.write('The custome is likely to churn.')
else:

    st.write("The custome is not likely to churn.")

