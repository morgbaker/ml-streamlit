import streamlit as st
import os
import sys

# Ensure DeepMoji is in the Python path
sys.path.append(os.path.join(os.getcwd(), 'DeepMoji'))

from deepmoji.model import DeepMoji  # Import the DeepMoji model

# Load your model
model = DeepMoji.load_model('DeepMoji/model.h5')  # Adjust according to the actual model file



