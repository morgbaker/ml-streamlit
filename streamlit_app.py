import sys
sys.path.append('/workspaces/ml-streamlit/torchMoji')
import streamlit as st
import json
import numpy as np
from torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from torchMoji.torchmoji.model_def import torchmoji_emojis


# A list of all the emojis the model might respond with.
# You'll need to fill in this list or load it from an appropriate source.
EMOJIS = [
    "😀", "😂", "😍", "😢", "😡", "👍", "🎉", "❤️", "🤔", "😎", 
    # Add all other emojis as needed...
]

# Specify the paths to the vocabulary and model weights files. 
vocab_file_path = 'torchMoji/model/vocabulary.json'
model_weights_path = 'torchMoji/model/pytorch_model.bin'

# Returns the indices of the k largest elements in array.
def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

class Emojize:
    def __init__(self):
        with open(vocab_file_path, 'r') as f:
            vocabulary = json.load(f)

        max_sentence_length = 100
        self.st = SentenceTokenizer(vocabulary, max_sentence_length)
        self.model = torchmoji_emojis(model_weights_path)

    def predict(self, text):
        if not isinstance(text, list):
            text = [text]

        tokenized, _, _ = self.st.tokenize_sentences(text)
        prob = self.model(tokenized)[0]
        emoji_ids = top_elements(prob, 1)
        emojis = list(map(lambda x: EMOJIS[x].strip(':'), emoji_ids))
        return emojis[0]

# Streamlit UI
st.set_page_config(page_title="Emoji Predictor", page_icon="🙂")
st.title("Emoji Predictor using Deep Learning! 😄")

# User input
user_input = st.text_area("Enter your text to predict the emoji:", height=200)

if st.button("Predict Emoji"):
    if user_input:
        emojize = Emojize()
        predicted_emoji = emojize.predict(user_input)
        st.success(f"The predicted emoji is: {predicted_emoji}")
    else:
        st.warning("Please enter some text for emoji prediction.")

# Footer
st.markdown("<br><hr><center>Made with ❤️ by Morgan</center><hr>", unsafe_allow_html=True)







