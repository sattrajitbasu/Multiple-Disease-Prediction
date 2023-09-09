# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 20:59:22 2023

@author: SATTRAJIT
"""

import pickle
import streamlit as st
import numpy as np
import pandas as pd
import statistics as stat
from streamlit_option_menu import option_menu

#Importing the datasets for Scaling the inputs
parkinsons_data = pd.read_csv('"D:/MultipleDiseasePrediction/Datasets/diabetes.csv"')
heart_data = pd.read_csv('"D:/MultipleDiseasePrediction/Datasets/heart_disease_data.csv"')
diabetes_dataset = pd.read_csv('"D:/MultipleDiseasePrediction/Datasets/diabetes.csv"')



# loading the saved models

diabetes_model_lr = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/diabetes_model.sav', 'rb'))
diabetes_model_svc = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/diabetes_model.sav', 'rb'))
diabetes_model_knn = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/diabetes_model.sav', 'rb'))
diabetes_model_dt = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/diabetes_model.sav', 'rb'))
diabetes_model_rf = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/diabetes_model.sav', 'rb'))


heart_disease_model_lr = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/heart_disease_model.sav', 'rb'))
heart_disease_model_svc = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/heart_disease_model.sav', 'rb'))
heart_disease_model_knn = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/heart_disease_model.sav', 'rb'))
heart_disease_model_dt = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/heart_disease_model.sav', 'rb'))
heart_disease_model_rf = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/heart_disease_model.sav', 'rb'))

parkinsons_model_lr = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/parkinsons_model.sav', 'rb'))
parkinsons_model_svc = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/parkinsons_model.sav', 'rb'))
parkinsons_model_knn = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/parkinsons_model.sav', 'rb'))
parkinsons_model_dt = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/parkinsons_model.sav', 'rb'))
parkinsons_model_rf = pickle.load(open('D:/MultipleDiseasePrediction/SavedModels/parkinsons_model.sav', 'rb'))



# sidebar to navigate

with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System', 
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Disease Prediction'],
                           
                           icons = ['activity', 'heart', 'person'],
                           
                            default_index=0)


# Diabetes Prediction Page
if(selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction System')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
        
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction1 = diabetes_model_lr.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction2 = diabetes_model_svc.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction3 = diabetes_model_knn.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction4 = diabetes_model_dt.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        diab_prediction5 = diabetes_model_rf.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        all_pred_diab = np.array([diab_prediction1[0], diab_prediction2[0], diab_prediction3[0], diab_prediction4[0], diab_prediction5[0]])
        
        # Most Predicted Value
        final_prediction_diab = stat.mode(all_pred_diab)
        
        if (final_prediction_diab == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
        
        
        
        
    
# Heart Disease Prediction Page
if(selected == 'Heart Disease Prediction'):
    # page title
    st.title('Heart Disease Prediction System')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    
    
    
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction1 = heart_disease_model_lr.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])
        heart_prediction2 = heart_disease_model_svc.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        heart_prediction3 = heart_disease_model_knn.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])
        heart_prediction4 = heart_disease_model_dt.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])
        heart_prediction5 = heart_disease_model_rf.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])
        
        all_pred_heart = np.array([heart_prediction1[0], heart_prediction2[0], heart_prediction3[0], heart_prediction4[0], heart_prediction5[0]])
        final_prediction_heart = stat.mode(all_pred_heart)
        
        if (final_prediction_heart == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
        
        
        
    
# Parkinsons Disease Prediction Page
if(selected == 'Parkinsons Disease Prediction'):
    # page title
    st.title('Parkinsons Disease Prediction System')
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
    
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction1 = parkinsons_model_lr.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        parkinsons_prediction2 = parkinsons_model_svc.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        parkinsons_prediction3 = parkinsons_model_knn.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        parkinsons_prediction4 = parkinsons_model_dt.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        parkinsons_prediction5 = parkinsons_model_rf.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        all_pred_parkin = np.array([parkinsons_prediction1[0], parkinsons_prediction2[0], parkinsons_prediction3[0], parkinsons_prediction4[0], parkinsons_prediction5[0]])
        
        final_prediction_parkin = stat.mode(all_pred_parkin)
        
        if (final_prediction_parkin == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)