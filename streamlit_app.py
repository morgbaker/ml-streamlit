import os
import sys
import numpy as np
import streamlit as st

# Add DeepMoji folder to system path
DEEP_MOJI_PATH = os.path.join(os.path.dirname(__file__), 'DeepMoji')
sys.path.append(DEEP_MOJI_PATH)

from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import load_specific_weights

# Load DeepMoji model
@st.cache_resource
def load_deepmoji():
    maxlen = 30
    batch_size = 32
    stokenizer = SentenceTokenizer(VOCAB_PATH, maxlen)

    # Load the pre-trained model
    model = deepmoji_emojis(maxlen, os.path.join('pretrained_weights', 'weights.h5'))
    load_specific_weights(model, os.path.join('pretrained_weights', 'weights.h5'), exclude_namespaces=['output_layer'])

    return stokenizer, model

stokenizer, model = load_deepmoji()

# Define function to get emoji predictions
def predict_emojis(text, top_k=5):
    tokenized, _, _ = stokenizer.tokenize_sentences([text])
    prob = model.predict(tokenized)[0]
    top_indices = np.argsort(prob)[-top_k:][::-1]
    emojis = [":{}:".format(index) for index in top_indices]  # Placeholder; replace with actual emoji mapping
    probabilities = [prob[i] for i in top_indices]
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

