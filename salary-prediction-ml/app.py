import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load the saved model
model = tf.keras.models.load_model('/content/salary_classifier.h5')

# Define categorical columns
categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'sex', 'native-country']
numerical_features = ['age']

# Input from user
st.title("Income Prediction App")

age = st.number_input("Age", min_value=17, max_value=75, step=1)
workclass = st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 
                                       'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education = st.selectbox("Education", ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 
                                       'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', 
                                       '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])
marital_status = st.selectbox("Marital Status", ['Married-civ-spouse', 'Divorced', 'Never-married', 
                                                 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Sales', 'Exec-managerial', 
                                         'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 
                                         'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 
                                         'Armed-Forces','Other-service'])
sex = st.selectbox("Sex", ['Male', 'Female'])

native_country = st.selectbox("Native Country", ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 
                                                 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 
                                                 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 
                                                 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 
                                                 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 
                                                 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 
                                                 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 
                                                 'Holand-Netherlands','others'])

# Prepare input
input_dict = {
    'age': age,
    'workclass': workclass,
    'education': education,
    'marital-status': marital_status,
    'occupation': occupation,
    'sex': sex,
    
    'native-country': native_country
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = encoder.fit_transform(input_df[categorical_features])

# Combine numerical + encoded categorical
final_input = np.concatenate([input_df[numerical_features].values, encoded_cats], axis=1)

# Predict
if st.button("Predict"):
    prediction = model.predict(final_input)
    if prediction[0][0] > 0.5:
        st.success("Prediction: Income > 50K")
    else:
        st.success("Prediction: Income <= 50K")
