import streamlit as st
import torch
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load DeepMoji model and tokenizer
@st.cache_resource
def load_deepmoji_model():
    # Assuming DeepMoji model is available on Hugging Face; replace with actual model details
    tokenizer = AutoTokenizer.from_pretrained("DeepMoji")
    model = AutoModelForSequenceClassification.from_pretrained("DeepMoji")
    return tokenizer, model

tokenizer, model = load_deepmoji_model()

# Define function to get emoji predictions
def predict_emojis(text, top_k=5):
    # Tokenize and prepare input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Predict emoji probabilities
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze().numpy()
    
    # Get top k emojis
    top_indices = np.argsort(probs)[-top_k:][::-1]
    emojis = [emoji.emojize(f":{index}:") for index in top_indices]  # Placeholder; replace with actual emoji mapping
    probabilities = [probs[i] for i in top_indices]
    
    return list(zip(emojis, probabilities))

# Streamlit UI
st.title("DeepMoji Emotion Analyzer")
user_text = st.text_input("Enter your text to analyze:")

if st.button("Analyze"):
    if user_text:
        predictions = predict_emojis(user_text)
        st.write("Top predicted emojis:")
        for emo, prob in predictions:
            st.write(f"{emo}: {prob:.2f}")
    else:
        st.warning("Please enter some text to analyze.")


