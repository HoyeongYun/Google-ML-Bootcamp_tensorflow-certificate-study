import tensorflow as tf
import numpy as np
import os
import wget

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if not os.path.exists('./irish-lyrics-eof.txt'):
    wget.download('https://storage.googleapis.com/tensorflow-1-public/course3/irish-lyrics-eof.txt')

data = open('./irish-lyrics-eof.txt').read()        # txt file 고대로 string 으로

corpus = data.lower().split('\n')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1             # text gen 에서는 oov 를 포함시키면 안되니깐 그자리 하나 padding 으로 준다고 생각

# sequences = tokenizer.texts_to_sequences(corpus) 아 text gen 에서는 이렇게 preprocessing 하면 안된다

input_sequences = []

for line in corpus:

    token_list = tokenizer.texts_to_sequences([line])

    for i in range(1, len(token_list)):

        input_sequences.append(token_list[:i+1])

max_sequences_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequences_len, padding='pre'))

xs, labels = input_sequences[:, :-1], input_sequences[-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# build model
embedding_dim = 100
lstm_dim = 150
learning_rate = 0.01

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(total_words, embedding_dim, input_length=max_sequences_len-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_dim)),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])

history = model.fit(xs, ys, epochs=100)

# text generating(model prediction)
seed_text = "help me obi-wan kinobi youre my only hope"

next_words = 100

for _ in range(next_words):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]

    token_list = pad_sequences([token_list], maxlen=max_sequences_len - 1, padding='pre')

    probabilities = model.predict(token_list)

    choice = np.random.choice([1, 2, 3])

    predicted = np.argsort(probabilities)[0][-choice]

    if predicted != 0:

        seed_text += ' ' + tokenizer.index_word[predicted]
