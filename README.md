# Disease Prediction System 🧑‍⚕️

An intelligent healthcare application that leverages machine learning models and Google's Generative AI to predict three major diseases (Diabetes, Heart Disease, and Parkinson's Disease) and provides personalized AI-powered medical recommendations.

## Overview

The Disease Prediction System is a comprehensive Streamlit-based web application that combines pre-trained machine learning models with Google's Gemini AI to provide accurate disease predictions and personalized healthcare suggestions. The system acts as a virtual health consultant, analyzing patient parameters and offering expert-level medical guidance for diabetes, heart disease, and Parkinson's disease management.

## Key Features

### 🔬 Multi-Disease Prediction
- **Diabetes Prediction**: Using 8 clinical parameters including glucose levels, BMI, and family history
- **Heart Disease Prediction**: Analyzing 13 cardiovascular parameters including cholesterol, blood pressure, and ECG results
- **Parkinson's Disease Prediction**: Evaluating 22 voice and speech parameters for neurological assessment

### 🤖 AI-Powered Medical Recommendations
- **Expert-Level Consultations**: AI acts as specialist doctors (diabetologist, cardiologist, neurologist)
- **Personalized Diet Plans**: Custom nutrition recommendations based on patient parameters
- **Treatment Suggestions**: Evidence-based treatment options and medication guidance
- **Lifestyle Modifications**: Comprehensive precautions and lifestyle recommendations

### 🖥️ User-Friendly Interface
- **Intuitive Navigation**: Clean sidebar menu with disease-specific sections
- **Responsive Design**: Wide layout optimized for all screen sizes
- **Real-time Validation**: Input validation with clear error messaging
- **Professional UI**: Medical-grade interface with appropriate icons and styling

### 🛡️ Medical Safety Features
- **Disclaimer Integration**: Clear warnings about AI-generated suggestions
- **Professional Consultation Reminders**: Emphasis on consulting healthcare professionals
- **Evidence-Based Responses**: AI recommendations based on medical best practices

## Prerequisites

### System Requirements
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for optimal performance)
- Internet connection for Google AI API access
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Required Accounts & API Keys
- **Google AI Studio Account**: Required for Gemini API access
- **Gemini API Key**: Obtain from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Pre-trained Models
The system requires three pre-trained machine learning models:
- `diabetes_model.sav` - Diabetes prediction model
- `heart_disease_model.sav` - Heart disease prediction model
- `parkinsons_model.sav` - Parkinson's disease prediction model

## Installation

### Method 1: Quick Setup
```bash
# Clone the repository
git clone https://github.com/ashrithvelisoju/Prediction-of-Disease-Outbreaks.git
cd Prediction-of-Disease-Outbreaks

# Create virtual environment
python -m venv disease_prediction_env
source disease_prediction_env/bin/activate  # On Windows: disease_prediction_env\Scripts\activate

# Install core dependencies
pip install streamlit scikit-learn pandas google-generativeai streamlit-option-menu
