import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu

# Set up the page configuration
st.set_page_config(
    page_title='Disease Prediction System',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load models
diabetes_model = pickle.load(open("Diabetes_model.sav", 'rb'))
heart_disease_model = pickle.load(open("Heart_model.sav", 'rb'))
parkinsons_model = pickle.load(open("parkinson_model.sav", 'rb'))
stroke_model = pickle.load(open('stroke_model(1).pkl', 'rb'))

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Disease Predictive System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinson Prediction', 'Stroke Prediction'],
        icons=['activity', 'heart', 'person', '🧠'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    col1,col2,col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of prehnancies')

    with col2:
        Glucose = st.text_input('Glucose level')

    with col3:
        BloodPressure = st.text_input('Blood pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
    with col2:
        Age = st.text_input('Age of the Person')


    # prediction
    dia_diagnosis = ''

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        
        if (diab_prediction[0] == 0):
            dia_diagnosis  = 'The person is not diabetic'
        else:
            dia_diagnosis = 'The person is diabetic'
            
    st.success(dia_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
    with col2:
        sex = st.number_input('Sex (1=Male, 0=Female)')
    with col3:
        cp = st.number_input('Chest Pain Types')
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
    with col1:
        restecg = st.number_input('Resting Electrocardiographic Results')
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.number_input('Exercise Induced Angina')
    with col1:
        oldpeak = st.number_input('ST Depression Induced by Exercise')
    with col2:
        slope = st.number_input('Slope of the Peak Exercise ST Segment')
    with col3:
        ca = st.number_input('Major Vessels Colored by Flourosopy')
    with col1:
        thal = st.number_input('Thal: 0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect')
        
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])                          
        
        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
    
# Parkinson's Prediction Page
if selected == "Parkinson Prediction":
    st.title("Parkinson's Disease Prediction using ML")

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
        


    parkinsons_diagnosis = ''


    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

# Stroke Prediction Page
if selected == "Stroke Prediction":
    st.header('🧠 STROKE PREDICTION MODEL')
    
    st.sidebar.write('Input Features:')
    age = st.sidebar.slider('Age:', 1, 100, 20)
    avg_glucose_level = st.sidebar.slider('Glucose Level', 1.0, 500.0, 70.0)
    bmi = st.sidebar.slider('BMI', 1.0, 100.0, 24.9)
    ever_married = st.radio("Are you married?", ('Yes', 'No'))
    gender = st.radio("What is your gender?", ('Male', 'Female'))
    work_type = st.radio("Which of the following best describes your work type?", ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    residence_type = st.radio("What is your residence type?", ('Urban', 'Rural'))
    smoking_status = st.radio("What is your smoking status?", ('formerly smoked', 'never smoked', 'smokes'))

    # Convert categorical variables to numerical format
    ever_married_indx = 1 if ever_married == "Yes" else 0
    gender_indx = 1 if gender == "Male" else 0

    work_type_indx = {
        "Private": 0,
        "Self-employed": 0,
        "Govt_job": 0,
        "children": 0,
        "Never_worked": 0,
    }
    work_type_indx[work_type] = 1

    residence_type_indx = 1 if residence_type == "Urban" else 0

    smoking_status_indx = {
        "formerly smoked": 0,
        "never smoked": 0,
        "smokes": 0,
    }
    smoking_status_indx[smoking_status] = 1

    data = {
        "age": [age],
        "avg_glucose_level": [avg_glucose_level],
        "bmi": [bmi],
        "gender_Male": [gender_indx],
        "ever_married_Yes": [ever_married_indx],
        "work_type_Govt_job": [work_type_indx["Govt_job"]],
        "work_type_Never_worked": [work_type_indx["Never_worked"]],
        "work_type_Private": [work_type_indx["Private"]],
        "work_type_Self-employed": [work_type_indx["Self-employed"]],
        "work_type_children": [work_type_indx["children"]],
        "Residence_type_Urban": [residence_type_indx],
        "smoking_status_formerly smoked": [smoking_status_indx["formerly smoked"]],
        "smoking_status_smokes": [smoking_status_indx["smokes"]]
    }

    test_df = pd.DataFrame(data)

    # Predict stroke probability
    pred_prob = stroke_model.predict_proba(test_df)[:, 1][0]

    st.subheader('Output')

    if pred_prob >= 0.5:
        st.markdown("<h1 style='font-size: 40px; color: red;'>The person might have a stroke.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='font-size: 40px; color: green;'>The person might not have a stroke.</h1>", unsafe_allow_html=True)

    st.metric('Predicted Probability of Having a Stroke:', f"{pred_prob:.2f}")
