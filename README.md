🧠 Next Word Predictor using LSTM

A deep learning based web application that predicts the next word in a sentence using a trained LSTM model built with TensorFlow/Keras.

📌 Project Overview

This project implements a Next Word Prediction system using:

Embedding Layer

Stacked LSTM Layers

Dense Softmax Output Layer

The model is trained on kaggle  data and deployed as a Streamlit web application.

🚀 Features

Predicts next word based on user input

Uses trained LSTM architecture

Tokenizer-based word indexing

Reverse word mapping for readable output

Simple and interactive UI using Streamlit

🏗 Model Architecture
Input(shape=(324,)) # 
Embedding(input_dim=4994, output_dim=200)
LSTM(150, return_sequences=True)
LSTM(100)
Dense(4994, activation='softmax')

Architecture Explanation

Embedding Layer → Converts word indices into dense vectors--->each words have 200 dim 

First LSTM Layer → Captures sequential dependencies

Second LSTM Layer → Learns deeper contextual representation

Dense Layer (Softmax) → Predicts probability of next word

## 📂 Project Structure

```
Word_Predictor/
│
├── app.py
├── word_predictor_model.h5
├── tokenizer.pickle
├── requirements.txt
└── README.md
```


⚙️ Installation & Setup
1️⃣ Create Conda Environment
conda create -n nextwordlstm python=3.10
conda activate nextwordlstm


⚠️ Recommended Python version: 3.10
TensorFlow 2.19 does not fully support Python 3.13.

2️⃣ Install Dependencies
pip install -r requirements.txt


Example requirements.txt:

tensorflow==2.19.0
streamlit
numpy
pickle-mixin


Note: Do NOT manually fix h5py version. TensorFlow will install compatible version automatically.

▶️ Run the Application
streamlit run app.py


App will open in browser at:

http://localhost:8501

🧩 How It Works

User enters text.

Text is tokenized using saved tokenizer.

Sequence is padded to fixed length (324).

Model predicts next word index.

Reverse mapping converts index → word.

Word is appended to input and displayed.

🧠 Training Details

Vocabulary Size: 4994 words

Sequence Length: 324

Embedding Dimension: 200

Optimizer: Adam

Loss Function: Categorical Crossentropy

📦 Model Saving
model.save("word_predictor_model.h5")


Tokenizer saved using:

pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))

🔮 Future Improvements

Add multi-word prediction

Add beam search instead of greedy selection

Deploy on Streamlit Cloud

Convert to TensorFlow SavedModel format

Add temperature sampling for creative text

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy

Streamlit

Environment:nextwordllstm

👨‍💻 Author

Upendra Chauhan
Deep Learning & NLP Enthusiast