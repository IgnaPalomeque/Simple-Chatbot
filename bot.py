import re
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pathlib import Path

print(f'Using Tensorflow version: {tf.__version__}')

# Parameters
data_fraction = 0.1 # Fraction of the data to use. Set to 1 to use all the data (will result in longer training times)
max_tok_lenght = 100 # Max lenght for a sequence
embedding_filters = 50 # Number of filters for the embedding layer
lstm_filters = 50 # Number of filters for the lstm layer
bsize = 64 # Size for the batch of data to feed the model
epochs = 10 # Number of epochs
train_model = True # For either training a new model or using an already trained one


# Path to files
path_to_dialogs =Path("./Dialogs")
path_to_lines = path_to_dialogs / "movie_lines.txt"
path_to_conversations = path_to_dialogs / "movie_conversations.txt"
save_path = Path("./pretrained-models")

# Check path
print(path_to_dialogs)
print(path_to_lines)
print(path_to_conversations)

# Creates a dictionary line_id:line
def load_lines(path):
    lines = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            part = line.strip().split(' +++$+++ ')
            if len(part) > 4:
                lines[part[0]] = part[4]
    return lines

# Creates an array with the connected movie lines
def load_conversations(path, lines):
    conversations = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for conv in file:
            part = conv.strip().split(' +++$+++ ')
            conv_ids = eval(part[-1])
            conversations.append([lines[cid] for cid in conv_ids if cid in lines])
    return conversations

lines = load_lines(path_to_lines)
conversations = load_conversations(path_to_conversations, lines)

# Creates an array with the input_text and output_text
pairs = []
for conv in conversations:
    for i in range(0, len(conv) - 1, 2):
        input_text = conv[i]
        output_text = conv[i + 1]
        pairs.append((input_text, output_text))

# Cleans text given
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
    text = re.sub(r"[.]+", ".", text)
    return text

# Changes the amount of total data to use
pairs = pairs[:int(len(pairs) * data_fraction)]

# Cleans input and output texts
input_texts = [clean_text(pair[0]) for pair in pairs]
output_texts = ["<start> " + clean_text(pair[1]) + " <end>" for pair in pairs]

pairs = list(zip(input_texts,output_texts))
random.shuffle(pairs)

input_texts = [pair[0] for pair in pairs]
output_texts = [pair[1] for pair in pairs]

'''max_tok_lenght = input_texts + output_texts
max_tok_lenght = sorted(max_tok_lenght, key=len)
max_tok_lenght = len(max_tok_lenght[-1])'''


# Tokenizes inputs and outputs
tokenizer = Tokenizer(num_words=None,filters='',oov_token='<OOV>')
tokenizer.fit_on_texts(input_texts + output_texts)
tokenized_inputs = tokenizer.texts_to_sequences(input_texts)
tokenized_outputs = tokenizer.texts_to_sequences(output_texts)
tokenized_tdata = tokenizer.texts_to_sequences(output_texts)
tokenized_tdata = [pair[1:] for pair in tokenized_outputs]


# Adds padding to target max sequence
tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=max_tok_lenght, padding='post', truncating='post')
tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=max_tok_lenght, padding='post', truncating='post')
tokenized_tdata = pad_sequences(tokenized_tdata, maxlen=max_tok_lenght, padding='post', truncating='post')

print(f'Shape of input: {tokenized_inputs.shape}')
print(f'Shape of output: {tokenized_outputs.shape}')
print(f'Shape of tdata: {tokenized_tdata.shape}')


# For testing
'''detoken_inputs = tokenizer.sequences_to_texts(tokenized_inputs)
detoken_outputs = tokenizer.sequences_to_texts(tokenized_outputs)
detoken_tdata = tokenizer.sequences_to_texts(tokenized_tdata)'''

vocab_size = len(tokenizer.word_index) + 1

# Definition of the encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size, embedding_filters)(encoder_inputs)
encoder_lstm = LSTM(lstm_filters, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Definition of the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size, embedding_filters)
decoder_input = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(lstm_filters, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_input, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Training the model
model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

if train_model:
    print('--------------------------------------')
    print('|You are about to train a new model!.|')
    print('--------------------------------------')
    model.fit(
    x=[tokenized_inputs,tokenized_outputs],
    y=tokenized_tdata,
    batch_size = bsize,
    epochs = epochs
)
    model.save(save_path)
    print('Your model has been saved to:')
    print(f'{save_path}')
else:
    print('--------------------------------')
    print('|Skipped training a new model!.|')
    print('--------------------------------')
    model = load_model(save_path)