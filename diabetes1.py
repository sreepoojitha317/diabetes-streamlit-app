import streamlit as st
import pandas as pd
import pickle 
import numpy as np

st.title('Classification Model')
st.write('Diabetes Yes or No?')

def create_page():

    st.sidebar.title('Enter Patient Details')

    preg = st.sidebar.number_input("Pregnancies",min_value = 0,max_value = 20)
    gluc = st.sidebar.slider('Glucose',min_value=0,max_value=200)
    bp = st.sidebar.slider('BloodPressure',min_value=0,max_value=140)
    skt = st.sidebar.slider('SkinThickness',min_value=0,max_value=100)
    ins = st.sidebar.number_input('Insulin',min_value = 0,max_value = 900)
    bmi = st.sidebar.slider('BMI',min_value=10.0,max_value= 60.0)
    dpf = st.sidebar.number_input('DiabetesPedigreeFunction',min_value = 0.0,max_value = 3.0)
    age = st.sidebar.slider('Age',min_value =10,max_value = 100)


    data = {
        'Pregnancies'	                :preg,
        'Glucose'	                :gluc,
        'BloodPressure'	                :bp,
        'SkinThickness'	                :skt,
        'Insulin'                       :ins,
        'BMI'	                        :bmi,
        'DiabetesPedigreeFunction'	:dpf,
        'Age'                           :age
        }
 
    inp_df =pd.DataFrame(data,index=[0])
    return inp_df
features = create_page()

if st.sidebar.button('Submit'):
   ##st.write(scaler.transform(features))
    loaded_model = pickle.load(open('diabetes_model.pkl','rb'))
    loaded_scaler = pickle.load(open("scaler.pkl", "rb"))
    st.write(features)
    features_scaled = loaded_scaler.transform(features)
    st.write(features_scaled)
    prob = loaded_model.predict_proba(features_scaled)[0][1]
   
    if prob >= 0.6:
        st.error("High Risk: Patient HAS Diabetes")
    else:
        st.success("Low Risk: Patient does NOT have Diabetes")








