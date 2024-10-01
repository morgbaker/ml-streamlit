import streamlit as st
import os
import sys

# Add the DeepMoji directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'DeepMoji'))

from deepmoji.model import DeepMoji  # Adjust the import based on your DeepMoji structure

# Load your DeepMoji model
model = DeepMoji.load_model('DeepMoji/model.h5')  # Replace with the actual model file name

# Create a simple Streamlit interface
st.title("DeepMoji Sentiment Analysis")

user_input = st.text_area("Enter text to analyze:")
if st.button("Analyze"):
    # Perform analysis using the DeepMoji model
    predictions = model.predict(user_input)  # Adjust this to fit your model's method
    st.write("Predictions:", predictions)


