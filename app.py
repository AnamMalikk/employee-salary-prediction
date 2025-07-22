import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np # Import numpy for handling potential unseen values

# Load the saved model
model = joblib.load("best_model.pkl")

# Load the original data to fit scalers and encoders
# This is crucial for consistent preprocessing
try:
    original_data = pd.read_csv("adult.csv")
except FileNotFoundError:
    st.error("Error: adult.csv not found. Please place the dataset file in the project root directory.")
    st.stop()


# --- Preprocessing Setup (Fit scalers and encoders on the original data) ---
temp_data_for_fitting = original_data.copy()

# Handle missing values for fitting
temp_data_for_fitting['occupation'] = temp_data_for_fitting['occupation'].replace({'?':'others'})
temp_data_for_fitting['native-country'] = temp_data_for_fitting['native-country'].replace({'?':'others'})
temp_data_for_fitting['workclass'] = temp_data_for_fitting['workclass'].replace({'?':'others'})

# Removing redundant categories for fitting
temp_data_for_fitting = temp_data_for_fitting[temp_data_for_fitting['workclass'] != 'Without-pay' ]
temp_data_for_fitting = temp_data_for_fitting[temp_data_for_fitting['workclass'] != 'Never-worked' ]
temp_data_for_fitting = temp_data_for_fitting[temp_data_for_fitting['education'] != '5th-6th' ]
temp_data_for_fitting = temp_data_for_fitting[temp_data_for_fitting['education'] != '1st-4th' ]
temp_data_for_fitting = temp_data_for_fitting[temp_data_for_fitting['education'] != 'Preschool' ]

# Drop redundant education column for fitting
temp_data_for_fitting.drop(columns=['education'], inplace=True)


# Fit Label Encoders
categorical_cols_to_encode = ['workclass', 'marital-status', 'occupation',
                              'relationship', 'race', 'gender', 'native-country']
encoders = {}
for col in categorical_cols_to_encode:
    enc = LabelEncoder()
    # Fit on combined data to handle all possible categories
    all_categories = original_data[col].unique().tolist()
    if '?' in all_categories:
      all_categories.remove('?')
      all_categories.append('others')
    # Ensure consistent order of classes
    all_categories.sort()
    enc.fit(all_categories)
    encoders[col] = enc

# Fit MinMaxScaler
numerical_cols_for_scaling = temp_data_for_fitting.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
scaler.fit(temp_data_for_fitting[numerical_cols_for_scaling])


# --- Preprocessing Function ---
def preprocess_input(input_df, encoders, scaler, numerical_cols):
    """
    Applies the same preprocessing steps to the input DataFrame from Streamlit.

    Args:
        input_df (pd.DataFrame): DataFrame containing the input features.
        encoders (dict): A dictionary of fitted LabelEncoders.
        scaler (MinMaxScaler): A fitted MinMaxScaler.
        numerical_cols (list): List of numerical column names used for scaling.

    Returns:
        pd.DataFrame: The preprocessed input data as a DataFrame.
    """
    processed_df = input_df.copy()

    # Handle missing values (assuming '?' is replaced by 'others' in the form or handled here)
    # For simplicity, we'll assume the form handles basic replacement or we handle it here
    for col in ['workclass', 'occupation', 'native-country']:
        if col in processed_df.columns:
            processed_df[col] = processed_df[col].replace({np.nan: 'others', '?': 'others'}) # Handle potential NaN from form as well

    # Apply Label Encoding
    for col, encoder in encoders.items():
        if col in processed_df.columns:
            # Ensure consistent order and handle potential unseen labels
            processed_df[col] = processed_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1) # Handle unseen with -1 or another strategy

    # Ensure numerical columns are present before scaling
    for col in numerical_cols:
        if col not in processed_df.columns:
             processed_df[col] = 0 # Add missing numerical columns with a default value if necessary

    # Apply Min-Max Scaling
    processed_df[numerical_cols] = scaler.transform(processed_df[numerical_cols])


    # Ensure all required columns are present in the processed_df in the correct order
    # This is important if the model expects features in a specific order
    # Get the list of columns the model was trained on (excluding the target)
    model_trained_cols = temp_data_for_fitting.drop(columns=['income']).columns.tolist()

    # Add any missing columns to the processed_df with a default value (e.g., 0)
    for col in model_trained_cols:
        if col not in processed_df.columns:
            processed_df[col] = 0

    # Reorder columns to match the training data
    processed_df = processed_df[model_trained_cols]


    return processed_df


# 🔹 Set page config
st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

# 🔹 Custom CSS for styling
st.markdown("""
    <style>
    /* Gradient background */
    body {
        background: linear-gradient(to right, #e0c3fc, #8ec5fc);
    }

    /* Main content area styling */
    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }

    /* Title style */
    h1 {
        color: #4B0082;
        font-weight: 700;
        text-shadow: 1px 1px 2px #fff;
    }

    /* Button style */
    .stButton > button {
        background: linear-gradient(to right, #FF512F, #DD2476);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 16px;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #DD2476, #FF512F);
        color: white;
    }

    /* Sidebar header */
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #fceabb, #f8b500);
    }

    /* Smaller tweaks */
    .stMarkdown h3 {
        color: #2e2e2e;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# App Title
st.title("💼 Employee Salary Classification App")
st.markdown("##### Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar Inputs (Update these to match the features used by your model)
# Refer to the columns in your 'data' DataFrame after preprocessing in the notebook
st.sidebar.header("📋 Input Employee Details")

# Get unique values for dropdowns from the original data (after replacing '?')
workclass_options = sorted(original_data['workclass'].replace({'?':'others'}).unique().tolist())
marital_status_options = sorted(original_data['marital-status'].unique().tolist())
occupation_options = sorted(original_data['occupation'].replace({'?':'others'}).unique().tolist())
relationship_options = sorted(original_data['relationship'].unique().tolist())
race_options = sorted(original_data['race'].unique().tolist())
gender_options = sorted(original_data['gender'].unique().tolist())
native_country_options = sorted(original_data['native-country'].replace({'?':'others'}).unique().tolist())
education_num_min = int(original_data['educational-num'].min())
education_num_max = int(original_data['educational-num'].max())


age = st.sidebar.slider("Age", int(original_data['age'].min()), int(original_data['age'].max()), 30) # Use original data min/max
workclass = st.sidebar.selectbox("Workclass", workclass_options)
fnlwgt = st.sidebar.number_input("Fnlwgt", int(original_data['fnlwgt'].min()), int(original_data['fnlwgt'].max()), 189664) # Use original data min/max, mean as default
educational_num = st.sidebar.slider("Educational Number", education_num_min, education_num_max, 10)
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
occupation = st.sidebar.selectbox("Occupation", occupation_options)
relationship = st.sidebar.selectbox("Relationship", relationship_options)
race = st.sidebar.selectbox("Race", race_options)
gender = st.sidebar.selectbox("Gender", gender_options)
capital_gain = st.sidebar.number_input("Capital Gain", int(original_data['capital-gain'].min()), int(original_data['capital-gain'].max()), 0)
capital_loss = st.sidebar.number_input("Capital Loss", int(original_data['capital-loss'].min()), int(original_data['capital-loss'].max()), 0)
hours_per_week = st.sidebar.slider("Hours per week", int(original_data['hours-per-week'].min()), int(original_data['hours-per-week'].max()), 40)
native_country = st.sidebar.selectbox("Native Country", native_country_options)


# Build input DataFrame
input_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}
input_df = pd.DataFrame([input_data])


# Show Input Data
st.write("### 🔎 Input Data")
st.dataframe(input_df)

# Predict Button
if st.button("Predict Salary Class"):
    # Preprocess the input data
    preprocessed_input_df = preprocess_input(input_df.copy(), encoders, scaler, numerical_cols_for_scaling)

    # Make a prediction using the best model
    prediction = model.predict(preprocessed_input_df)

    st.success(f"✅ Prediction: **{prediction[0]}**")

# Divider
st.markdown("---")

# Batch Prediction Section (Update this to also use the preprocessing function)
st.markdown("### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", batch_data.head())

    # Preprocess the batch data
    preprocessed_batch_data = preprocess_input(batch_data.copy(), encoders, scaler, numerical_cols_for_scaling)

    # Make batch predictions
    batch_preds = model.predict(preprocessed_batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.dataframe(batch_data.head())

    # Download CSV
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
