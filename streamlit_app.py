import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the TorchMoji model and tokenizer from Hugging Face
model_name = "huggingface/torchmoji"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

st.title("TorchMoji Sentiment Analysis")

user_input = st.text_area("Enter text to analyze:")
if st.button("Analyze"):
    # Tokenize the input text
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.softmax(logits, dim=1).numpy()

    # Display predictions (for example, the predicted classes)
    st.write("Predictions:", predictions)




