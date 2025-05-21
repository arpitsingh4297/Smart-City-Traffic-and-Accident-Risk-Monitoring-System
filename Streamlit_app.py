import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Accident Severity Prediction (New)", layout="wide")

# Determine the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
st.write(f"Script Directory: {script_dir}")
st.write(f"Current Working Directory: {os.getcwd()}")

# Function to load a file with error handling
def load_file(filename, directory):
    file_path = os.path.join(directory, filename)
    try:
        obj = joblib.load(file_path)
        st.write(f"Successfully loaded {filename}")
        return obj
    except FileNotFoundError:
        st.error(f"File '{filename}' not found at {file_path}. Please ensure the file exists in the script directory.")
        return None
    except Exception as e:
        st.error(f"Error loading {filename}: {str(e)}")
        return None

# Load saved objects
files_to_load = {
    'best_model.pkl': 'model',
    'scaler.pkl': 'scaler',
    'selected_features.pkl': 'selected_features',
    'label_encoder.pkl': 'le',
    'encoders.pkl': 'encoders',
    'feature_options.pkl': 'feature_options'
}

loaded_objects = {}
for filename, obj_name in files_to_load.items():
    loaded_obj = load_file(filename, script_dir)
    if loaded_obj is None:
        st.error(f"Failed to load {filename}. Cannot proceed without all required files.")
        st.stop()
    loaded_objects[obj_name] = loaded_obj

# Assign loaded objects
model = loaded_objects['model']
scaler = loaded_objects['scaler']
selected_features = loaded_objects['selected_features']
le = loaded_objects['le']
encoders = loaded_objects['encoders']
feature_options = loaded_objects['feature_options']

# Define feature categories
categorical_cols = ['State Name', 'Vehicle Type Involved', 'Weather Conditions', 'Lighting Conditions', 'Accident Location Details']
numerical_cols = ['Number of Vehicles Involved', 'Number of Casualties', 'Number of Fatalities', 'Speed Limit (km/h)', 'Driver Age']

# Verify that feature_options match encoders
for col in categorical_cols:
    if col in encoders and col in feature_options:
        trained_classes = list(encoders[col].classes_)
        app_classes = feature_options.get(col, [])
        if set(trained_classes) != set(app_classes):
            st.warning(f"Categories for '{col}' in feature_options ({app_classes}) do not match training data ({trained_classes}). Using training data categories.")
            feature_options[col] = trained_classes

# Severity mapping
severity_mapping = {'Fatal': 0, 'Minor': 1, 'Serious': 2}
inverse_severity_mapping = {v: k for k, v in severity_mapping.items()}

# Streamlit App Layout
st.title("🚗 Accident Severity Prediction App (New Version)")
st.markdown("""
This app predicts the severity of road accidents in India based on input features. 
Enter the details below, and the model will predict whether the accident is **Fatal**, **Minor**, or **Serious**.
""")

# Sidebar for user inputs
st.sidebar.header("Input Accident Details")
user_inputs = {}

# Collect user inputs for selected features in the correct order
for feature in selected_features:
    if feature in categorical_cols:
        user_inputs[feature] = st.sidebar.selectbox(
            f"Select {feature}",
            options=feature_options.get(feature, []),
            key=feature
        )
    elif feature in numerical_cols:
        if feature == 'Number of Vehicles Involved':
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature} (1-10)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key=feature
            )
        elif feature == 'Number of Casualties':
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature} (0-20)",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
                key=feature
            )
        elif feature == 'Number of Fatalities':
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature} (0-10)",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                key=feature
            )
        elif feature == 'Speed Limit (km/h)':
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature} (10-120 km/h)",
                min_value=10,
                max_value=120,
                value=50,
                step=5,
                key=feature
            )
        elif feature == 'Driver Age':
            user_inputs[feature] = st.sidebar.number_input(
                f"{feature} (18-100 years)",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                key=feature
            )

# Main content
st.header("Prediction")
if st.button("Predict Accident Severity"):
    try:
        # Input validation
        if user_inputs['Number of Fatalities'] > user_inputs['Number of Casualties']:
            st.error("Number of Fatalities cannot exceed Number of Casualties.")
            st.stop()

        # Create input dataframe with features in the correct order
        input_df = pd.DataFrame([user_inputs])[selected_features]

        # Debug: Display input DataFrame
        st.write("Input DataFrame (before encoding):")
        st.write(input_df)

        # Encode categorical features using saved encoders
        for col in categorical_cols:
            if col in selected_features:
                try:
                    # Check if the input value is in the encoder's classes
                    if user_inputs[col] not in encoders[col].classes_:
                        st.error(f"Value '{user_inputs[col]}' for '{col}' not found in training data. Please select a valid category: {list(encoders[col].classes_)}")
                        st.stop()
                    input_df[col] = encoders[col].transform([user_inputs[col]])[0]
                except Exception as e:
                    st.error(f"Error encoding '{col}': {str(e)}. Ensure the selected value matches training data categories.")
                    st.stop()

        # Scale numerical features
        numerical_features = [col for col in numerical_cols if col in selected_features]
        if numerical_features:
            try:
                # Debug: Display numerical features being scaled
                st.write("Numerical Features to Scale:", numerical_features)
                # Debug: Display the columns the scaler was fitted on (if available)
                if hasattr(scaler, 'feature_names_in_'):
                    st.write("Scaler Expected Features:", list(scaler.feature_names_in_))
                input_df[numerical_features] = scaler.transform(input_df[numerical_features])
            except Exception as e:
                st.error(f"Error scaling numerical features: {str(e)}. Ensure numerical inputs are valid.")
                st.stop()

        # Debug: Display encoded and scaled DataFrame
        st.write("Input DataFrame (after encoding and scaling):")
        st.write(input_df)

        # Verify feature names and order
        if list(input_df.columns) != selected_features:
            st.error(f"Feature mismatch. Expected: {selected_features}, Got: {list(input_df.columns)}")
            st.stop()

        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]

        # Display results
        st.success(f"Predicted Accident Severity: **{inverse_severity_mapping[prediction]}**")
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Severity': list(severity_mapping.keys()),
            'Probability': [f"{prob:.2%}" for prob in probabilities]
        })
        st.table(prob_df)

        # Add note about prediction confidence
        max_prob = max(probabilities)
        if max_prob < 0.5:
            st.warning(f"The model's confidence in this prediction is relatively low ({max_prob:.2%}). The probabilities for all severity levels are close, indicating uncertainty. Consider re-training the model with additional data or features to improve accuracy. The current model's F1-score is 0.3478, which is low.")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}. Please check input values and ensure they match the training data format.")

# Model Insights
st.header("Model Insights")
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    plt.title('Feature Importance')
    st.pyplot(fig)
    plt.close(fig)
else:
    st.write("Feature importance not available for this model (Logistic Regression). Below are the top features selected by RFE:")
    st.write(selected_features)
    
    # Display Logistic Regression coefficients
    if hasattr(model, 'coef_'):
        coeffs = pd.DataFrame(
            model.coef_,
            index=[f"Class {inverse_severity_mapping[i]}" for i in range(len(severity_mapping))],
            columns=selected_features
        )
        st.subheader("Logistic Regression Coefficients")
        st.write(coeffs)

# Business Insights
st.header("Business Insights and Recommendations")
st.markdown("""
Based on the model's analysis, here are key insights and recommendations:
- **Weather Conditions**: Foggy and Rainy conditions are strongly associated with Fatal accidents. Implement real-time weather alerts in traffic management systems.
- **Speed Limits**: High speed limits correlate with severe accidents. Enforce stricter speed controls on National Highways.
- **Risk Profiles**: Clusters indicate distinct risk profiles. Target high-risk clusters (e.g., Foggy weather, Wet roads) for preventive measures like enhanced signage or police presence.
- **Real-Time Deployment**: Deploy this model with IoT traffic sensors to predict accident severity in real-time and trigger preventive actions.
""")

# Footer
st.markdown("---")
st.write("Built with Streamlit | Model trained on Indian accident dataset | © 2025 Accident Prediction System")