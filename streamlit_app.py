import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# prototype title
st.title("Data Viewing and Predictive Modeling App")

# sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Data View", "Predictive Modeling"])

# upload dataset
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

# reads df only if file is submitted
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # data view tab
    if page == "Data View":
        st.header("Data View")
        
        # displays dataset
        st.write("Dataset Overview:")
        st.dataframe(df)
        
        # shows basic stats of dataset
        st.write("Dataset Statistics:")
        st.write(df.describe())
        
        # feature selection
        selected_features = st.multiselect("Select features for EDA charts:", df.columns.tolist())
        
        if selected_features:
            # pairplot
            st.write("Pairplot of Selected Features:")
            sns.pairplot(df[selected_features])
            st.pyplot()

            # correlation heatmap
            st.write("Correlation Heatmap:")
            corr = df[selected_features].corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            st.pyplot()
    
    # predictive modeling tab
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

