import os.path
import tensorflow
import numpy as np
import wget
import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if not os.path.exists('./sarcasm.json'):
    wget.download('https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json')

with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)            # dictionary 들을 담고 있는 list

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
