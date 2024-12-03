import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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