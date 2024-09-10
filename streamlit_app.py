import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Set the title of the Streamlit app
st.title("Data Viewing and Predictive Modeling App")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data View", "Predictive Modeling"])

# Upload dataset
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    
    # Tab 1: Data View
    if page == "Data View":
        st.header("Data View")
        
        # Display the dataset
        st.write("Dataset Overview:")
        st.dataframe(df)
        
        # Show dataset statistics
        st.write("Dataset Statistics:")
        st.write(df.describe())
        
        # Feature selection for EDA
        selected_features = st.multiselect("Select features for EDA charts:", df.columns.tolist())
        
        if selected_features:
            # Pairplot for selected features
            st.write("Pairplot of Selected Features:")
            sns.pairplot(df[selected_features])
            st.pyplot()

            # Correlation heatmap
            st.write("Correlation Heatmap:")
            corr = df[selected_features].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            st.pyplot()
    
    # Tab 2: Predictive Modeling
    elif page == "Predictive Modeling":
        st.header("Predictive Modeling")

        if 'df' not in locals():
            st.write("Please upload a CSV file to build a predictive model.")
        else:
            # Feature and target selection
            st.write("Select features and target for the model:")
            features = st.multiselect("Select features:", df.columns.tolist())
            target = st.selectbox("Select target:", df.columns.tolist())

            if len(features) > 0 and target:
                # Model selection
                model_type = st.selectbox("Select model type:", ["Linear Regression", "Ridge Regression", "Lasso Regression"])

                # Hyperparameters
                if model_type in ["Ridge Regression", "Lasso Regression"]:
                    alpha = st.slider("Select regularization strength (alpha):", 0.01, 10.0, 1.0)

                # Prepare data for modeling
                X = df[features]
                y = df[target]

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the model
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso Regression":
                    model = Lasso(alpha=alpha)

                model.fit(X_train, y_train)

                # Predict on the test set
                y_pred = model.predict(X_test)

                # Show model evaluation metrics
                st.write("Model Evaluation:")
                st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
                st.write(f"R-squared: {r2_score(y_test, y_pred):.4f}")

                # Plot residuals
                st.write("Residuals Plot:")
                residuals = y_test - y_pred
                plt.figure(figsize=(10, 6))
                sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
                plt.xlabel("Predicted Values")
                plt.ylabel("Residuals")
                plt.title("Residuals vs. Predicted Values")
                st.pyplot()
                
                # Display predictions vs actual values
                result_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write("Predictions vs Actual:")
                st.write(result_df)
            else:
                st.write("Please select the features and target for the model.")
else:
    st.write("Please upload a CSV file to begin.")
