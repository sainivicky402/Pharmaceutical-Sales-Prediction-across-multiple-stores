import streamlit as st
import pandas as pd
import joblib

# Load the model once at the beginning
@st.cache_resource
def load_model(path):
    """Load the pre-trained Random Forest model."""
    return joblib.load(path)

# Predict function
def make_prediction(model, data):
    """Generate predictions using the loaded model."""
    return model.predict(data)

# Main app function
def main():
    st.title("Store Prediction App")

    # Input section for store ID
    store_id = st.text_input("Enter Store ID")

    # File uploader for the CSV file
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type="csv")

    # Button to trigger prediction
    if st.button("Predict"):
        if store_id and uploaded_file:
            try:
                # Read the uploaded file and prepare data
                data = pd.read_csv(uploaded_file)

                # Load the model and make predictions
                model = load_model('Random_forest_model_11-14-2024-14-16-52-00.pkl')
                predictions = make_prediction(model, data)

                # Display results
                st.write(f"Predictions for Store ID {store_id}:")
                st.write(predictions)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
        else:
            st.warning("Please provide both a Store ID and a CSV file.")

# Run the app
if __name__ == "__main__":
    main()
