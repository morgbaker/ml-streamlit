import streamlit as st
import torch
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load DeepMoji model and tokenizer
@st.cache_resource
def load_deepmoji_model():
    tokenizer = AutoTokenizer.from_pretrained("DeepMoji")
    model = AutoModelForSequenceClassification.from_pretrained("DeepMoji")
    return tokenizer, model

tokenizer, model = load_deepmoji_model()

# Define function to get emoji predictions
def predict_emojis(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squee

