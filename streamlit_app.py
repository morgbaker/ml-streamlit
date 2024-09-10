import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Set the title of the Streamlit app
st.title("Data Viewing and Predictive Modeling App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data View", "Predictive Modeling"])

# Tab 1: Data View
if page == "Data View":
    st.header("Data View")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the dataset
        st.write("Dataset Overview:")
        st.dataframe(df)
        
        # Show dataset statistics
        st.write("Dataset Statistics:")
        st.write(df.describe())
        
        # Show column information
        st.write("Column Information:")
        st.write(df.info())
    else:
        st.write("Please upload a CSV file to view the data.")

# Tab 2: Predictive Modeling
elif page == "Predictive Modeling":
    st.header("Predictive Modeling")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload a CSV file for modeling", type="csv", key='2')

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the dataset
        st.write("Dataset Overview:")
        st.dataframe(df)
        
        # Feature and target selection
        st.write("Select features and target for the model:")
        features = st.multiselect("Select features:", df.columns.tolist())
        target = st.selectbox("Select target:", df.columns.tolist())

        if len(features) > 0 and target:
            # Prepare data for modeling
            X = df[features]
            y = df[target]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict on the test set
            y_pred = model.predict(X_test)

            # Show model evaluation metrics
            st.write("Model Evaluation:")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

            # Display predictions vs actual values
            result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.write("Predictions vs Actual:")
            st.write(result_df)

        else:
            st.write("Please select the features and target for the model.")
    else:
        st.write("Please upload a CSV file to build a predictive model.")
