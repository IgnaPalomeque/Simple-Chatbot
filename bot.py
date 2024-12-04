import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input
from tensorflow.keras.models import Model
from pathlib import Path

ROOT_DIR = Path(__file__).parent

# Specifying path to text files
lines_path = ROOT_DIR / "Dialogs/movie_lines.txt"
conv_path = ROOT_DIR / "Dialogs/movie_conversations.txt"

# Loads the lines file and returns a dictionary for each line
def load_lines(path):
    lines = {}
    with open(path, 'r',encoding='utf-8',errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) > 4:
                lines[parts[0]] = parts[4]
    return lines

# Loads the conversations files and return an array for the conversations
def load_conversations(path, lines):
    conversations = []
    with open(path, 'r',encoding='utf-8',errors='ignore') as file:
        for line in file:
            parts = line.strip().split(" +++$+++ ")
            conv_ids = eval(parts[-1])
            conversations.append([lines[cid] for cid in conv_ids if cid in lines])
    return conversations

lines = load_lines(lines_path)
conversations = load_conversations(conv_path, lines)

# Creates an array to store each questions and its response
pairs = []
for conv in conversations:
    for i in range(len(conv) - 1):
        input_text = conv[i]
        target_text = conv[i + 1]
        pairs.append((input_text, target_text))

# Clean text given
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = re.sub(r"[.]+", ".", text)
    return text

# Sets different arrays for questions and answers and cleans them
input_texts = [clean_text(pair[0]) for pair in pairs]
target_texts = ["<start> " + clean_text(pair[1]) + " <end>" for pair in pairs]

# Creates a dicctionary of word with its own number
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_texts + target_texts)

# Replaces each word with the predefined number
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# Adds padding to target the max sequence length
max_seq_length = max([len(seq) for seq in input_sequences + target_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
target_sequences = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')

vocab_size = len(tokenizer.word_index) + 1

# Definition of the encoder
encodoer_inputs = Input(shape=(max_seq_length,))
encoder_embedding = Embedding(vocab_size, 256)(encodoer_inputs)
encoder_lstm, state_h, state_c, = LSTM(256, return_state=True)(encoder_embedding)
encoder_states = [state_h,state_c]

# Definition of the decoder
decoder_inputs = Input(shape=(max_seq_length,))
decoder_embedding = Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=False)
decoder_outputs = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encodoer_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')
model.summary()