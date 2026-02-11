import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---- Config ----
st.set_page_config(page_title="Next Word Predictor", page_icon="🔮")

st.title("🔮 Next Word Prediction App")
st.write("Enter a sentence and predict the next word.")

# ---- Load Model ----
@st.cache_resource # decorator to cache the model loading=Kisi heavy cheez ko baar-baar reload hone se bachana.
def load_model():
    return tf.keras.models.load_model("word_predictor_model.h5", compile=False)

model = load_model()

# ---- Load Tokenizer ----
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()

# Reverse mapping
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# SAME as training
max_sequence_len = 324  

# ---- Input ----
user_input = st.text_input("Type your text here")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([user_input])

        # Pad properly (IMPORTANT: -1 for next word model)
        padded = pad_sequences(sequence, maxlen=max_sequence_len-1, padding='pre')

        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_index = np.argmax(prediction)

        # Convert index to word
        predicted_word = reverse_word_index.get(predicted_index, "")

        if predicted_word:
            st.success(f"Next word: **{predicted_word}**")
        else:
            st.error("Prediction not found in vocabulary.")
