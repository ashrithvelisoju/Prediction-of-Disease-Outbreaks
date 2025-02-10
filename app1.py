import os
import sklearn
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai

# Set page configuration
st.set_page_config(page_title="Prediction of Disease Outbreaks",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Configure Google Gemini API
GOOGLE_API_KEY = "AIzaSyAfDHl49CeXaIKc-j1pEDw1lJe6hAsKKj8"  # Store your API key in Streamlit secrets
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Function to get AI suggestions
def get_ai_suggestions(disease_type, parameters):
    if disease_type == "diabetes":
        prompt = f"""
        act as an expert diabetologist,field of pharmacology , virtual nutritionist and provide personalized healthcare suggestions for diabetes management.
        Based on the following parameters for a patient:
        - Pregnancies: {parameters[0]}
        - Glucose Level: {parameters[1]}
        - Blood Pressure: {parameters[2]}
        - Skin Thickness: {parameters[3]}
        - Insulin Level: {parameters[4]}
        - BMI: {parameters[5]}
        - Diabetes Pedigree: {parameters[6]}
        - Age: {parameters[7]}

        Provide comprehensive healthcare suggestions in the following format:

        1. Type of Diabete:
        [Mention the type of diabetes based on the parameters provided and mention name of the disease]

        2. Precautions:
        [List key precautions]

        3. Recommended Diet:
        - Vegetables: [List 5-6 best vegetables]
        - Fruits: [List 4-5 suitable fruits]
        - Proteins: [List protein sources]
        - Grains: [List recommended grains]
        - Foods to Avoid: [List foods to avoid]

        4. Meal Timing:
        [Provide basic meal timing suggestions]

        5. Sample Daily Meal Plan:
        - Breakfast:
        - Mid-morning Snack:
        - Lunch:
        - Evening Snack:
        - Dinner:

        6. Treatment Options:
        [List treatment options]

        7. Possible Medications to Discuss with Doctor:
        [List potential medications]

        Note: Keep the response concise but informative.
        """
    
    elif disease_type == "heart":
        prompt = f"""
        act as an expert cardiologist,field of pharmacology ,virtual nutritionist and provide personalized healthcare suggestions for heart disease management.
        Based on the following heart health parameters:
        - Age: {parameters[0]}
        - Blood Pressure: {parameters[3]}
        - Cholesterol: {parameters[4]}
        - Max Heart Rate: {parameters[7]}
        - Exercise Angina: {parameters[8]}
        - ST Depression: {parameters[9]}

        Provide comprehensive healthcare suggestions in the following format:

        1. Precautions:
        [List key precautions]

        2. Heart-Healthy Diet:
        - Vegetables: [List 5-6 heart-healthy vegetables]
        - Fruits: [List 4-5 suitable fruits]
        - Proteins: [List lean protein sources]
        - Grains: [List recommended whole grains]
        - Foods to Avoid: [List foods to avoid]

        3. Meal Planning:
        [Provide basic meal planning guidelines]

        4. Sample Heart-Healthy Daily Menu:
        - Breakfast:
        - Mid-morning Snack:
        - Lunch:
        - Evening Snack:
        - Dinner:

        5. Possible Medications to Discuss with Doctor:
        [List potential medications]

        Note: Keep the response concise but informative.
        """
    
    elif disease_type == "parkinsons":
        prompt = f"""
        act as an expert neurologist,field of pharmacology ,virtual nutritionist and provide personalized healthcare suggestions for Parkinson's disease management.
        Based on the following Parkinson's disease parameters:
        - Fundamental Frequency: {parameters[0]}
        - Jitter Percentage: {parameters[3]}
        - Shimmer: {parameters[8]}
        - NHR: {parameters[14]}
        - HNR: {parameters[15]}

        Provide comprehensive healthcare suggestions in the following format:

        1. Precautions:
        [List key precautions]

        2. Recommended Diet for Parkinson's:
        - Vegetables: [List 5-6 beneficial vegetables]
        - Fruits: [List 4-5 suitable fruits]
        - Proteins: [List protein sources]
        - Grains: [List recommended grains]
        - Foods to Avoid: [List foods to avoid]

        3. Dietary Tips:
        [Provide specific dietary considerations for Parkinson's]

        4. Sample Daily Menu:
        - Breakfast:
        - Mid-morning Snack:
        - Lunch:
        - Evening Snack:
        - Dinner:

        5. Possible Medications to Discuss with Doctor:
        [List potential medications]

        Note: Keep the response concise but informative.
        """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Prediction of Disease Outbreaks System by Ashrith Velisoju',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Helper function to validate inputs
def validate_inputs(inputs):
    try:
        return [float(x) for x in inputs if x.strip() != ""], None
    except ValueError:
        return None, "Please ensure all inputs are filled and are valid numbers."

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

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

    # Create two columns for buttons
    button_col1, button_col2 = st.columns(2)

    # creating a button for Prediction
    if button_col1.button('Diabetes Test Result'):
        user_input, error = validate_inputs([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if error:
            st.error(error)
        elif len(user_input) < 8:
            st.error("Please fill all the fields.")
        else:
            # Define feature names for diabetes prediction
            diabetes_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            import pandas as pd
            input_df = pd.DataFrame([user_input], columns=diabetes_features)
            diab_prediction = diabetes_model.predict(input_df)
            diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'
            st.success(diab_diagnosis)

    # Button for AI suggestions
    if button_col2.button('Get AI Suggestions'):
        user_input, error = validate_inputs([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        if error:
            st.error(error)
        elif len(user_input) < 8:
            st.error("Please fill all the fields.")
        else:
            with st.spinner('Getting AI suggestions...'):
                suggestions = get_ai_suggestions("diabetes", user_input)
                st.markdown("### AI-Generated Suggestions")
                st.markdown(suggestions)
                st.warning("Note: These are AI-generated suggestions. Always consult with a healthcare professional for medical advice.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

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

    # Create two columns for buttons
    button_col1, button_col2 = st.columns(2)

    # creating a button for Prediction
    if button_col1.button('Heart Disease Test Result'):
        user_input, error = validate_inputs([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        if error:
            st.error(error)
        elif len(user_input) < 13:
            st.error("Please fill all the fields.")
        else:
            # Define feature names for heart disease prediction
            heart_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            import pandas as pd
            input_df = pd.DataFrame([user_input], columns=heart_features)
            heart_prediction = heart_disease_model.predict(input_df)
            heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'
            st.success(heart_diagnosis)

    # Button for AI suggestions
    if button_col2.button('Get AI Suggestions'):
        user_input, error = validate_inputs([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        if error:
            st.error(error)
        elif len(user_input) < 13:
            st.error("Please fill all the fields.")
        else:
            with st.spinner('Getting AI suggestions...'):
                suggestions = get_ai_suggestions("heart", user_input)
                st.markdown("### AI-Generated Suggestions")
                st.markdown(suggestions)
                st.warning("Note: These are AI-generated suggestions. Always consult with a healthcare professional for medical advice.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
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

    # code for Prediction
    parkinsons_diagnosis = ''

    # Create two columns for buttons
    button_col1, button_col2 = st.columns(2)

    # creating a button for Prediction    
    if button_col1.button("Parkinson's Test Result"):
        user_input, error = validate_inputs([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        if error:
            st.error(error)
        elif len(user_input) < 22:
            st.error("Please fill all the fields.")
        else:
            # Define feature names for parkinsons prediction
            parkinsons_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                                 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                                 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                                 'spread1', 'spread2', 'D2', 'PPE']
            import pandas as pd
            input_df = pd.DataFrame([user_input], columns=parkinsons_features)
            parkinsons_prediction = parkinsons_model.predict(input_df)
            parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"
            st.success(parkinsons_diagnosis)

    # Button for AI suggestions
    if button_col2.button('Get AI Suggestions'):
        user_input, error = validate_inputs([fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE])
        if error:
            st.error(error)
        elif len(user_input) < 22:
            st.error("Please fill all the fields.")
        else:
            with st.spinner('Getting AI suggestions...'):
                suggestions = get_ai_suggestions("parkinsons", user_input)
                st.markdown("### AI-Generated Suggestions")
                st.markdown(suggestions)
                st.warning("Note: These are AI-generated suggestions. Always consult with a healthcare professional for medical advice.")