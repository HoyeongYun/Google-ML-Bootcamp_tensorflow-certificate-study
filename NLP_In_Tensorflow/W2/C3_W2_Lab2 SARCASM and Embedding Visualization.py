import os
import wget
import json
import numpy as np
import tensorflow as tf
import io

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

if not os.path.exists('./sarcasm.json'):
    url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
    wget.download(url)

with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_size = 20000

sentences_training = sentences[:training_size]
labels_training = labels[:training_size]

sentences_testing = sentences[training_size:]
labels_testing = labels[training_size:]

vocab_size = 10000
max_length = 32
embedding_dim = 16

trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences_training)

sequences_training = tokenizer.texts_to_sequences(sentences_training)
padded_training = pad_sequences(sequences_training, maxlen=max_length, truncating=trunc_type)

sequences_testing = tokenizer.texts_to_sequences(sentences_testing)
padded_testing = pad_sequences(sequences_testing, maxlen=max_length, truncating=trunc_type)

training_labels = np.array(labels_training)
testing_labels = np.array(labels_testing)

#### Global Max Pooling 1D  -> Global 이므로 2d일땐 한면, 1d 일땐 time 차원을 없애고 channels, embedding 차원만 남음
gap1d_layer = tf.keras.layers.GlobalAveragePooling1D()

sample_array = np.array([[[10,2],[1,3],[1,1]]])

print(f'shape of sample_array = {sample_array.shape}')          # (1, 3, 2)
print(f'sample array: {sample_array}')

output = gap1d_layer(sample_array)

print(f'output shape of gap1d_layer: {output.shape}')           # (1, 2)
print(f'output array of gap1d_layer: {output.numpy()}')

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(padded_training, labels_training, epochs=100, validation_data=(padded_testing, labels_testing))

# Word, Embedding 정보를 포함한 tsv파일 생성
out_m = io.open('meta.tsv', 'w', encoding='utf-8') # word 정보
out_v = io.open('vecs.tsv', 'w', encoding='utf-8') # Embedding 정보

embedding_weight = model.layers[0].get_weights()[0]         # (vocab_size, embedding)

for word_num in range(1, vocab_size):

    word_name = tokenizer.index_word[word_num]
    out_m.write(word_name + '\n')

    word_embedding = embedding_weight[word_num]
    out_v.write('\t'.join([str(x) for x in word_embedding]) + '\n')

out_m.close()
out_v.close()

