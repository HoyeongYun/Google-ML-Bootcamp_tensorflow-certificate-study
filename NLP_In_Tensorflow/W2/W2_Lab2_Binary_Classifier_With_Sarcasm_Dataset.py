import wget
import json
import io
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

if not os.path.exists('./sarcasm.json'):
    url = 'https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json'
    wget.download(url)
    print('???')

with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)

# datastore [{}, {}, {}]
# print(datastore)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_size = 20000
vocab_size = 10000
max_length = 32
embedding_dim = 16

# preprocess
# sentences에 있는 data를 train, val로 나누기
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[:training_size]
testing_labels = labels[training_size:]

trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# average pooling
gap1d_layer = tf.keras.layers.GlobalAveragePooling1D()
sample_array = np.array([[[10, 2],
                         [1, 3],
                         [1, 1]]])      # shape -> (1, 3, 2)       height width channel

print(f'shape of sample_array = {sample_array.shape}')
print(f'sample array = {sample_array}')

output = gap1d_layer(sample_array)      # 이렇게 layer 한개에 바로 input을 넣으면 return 은 tf.tensor

print(f'output shape of gap1d_layer = {output.shape}')          # (1, 2) (앞면은 하나로, channel)
print(f'output array of gap1d_layer = {output.numpy()}')

# build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

history = model.fit(training_padded,
                    training_labels,
                    validation_data=(testing_padded, testing_labels),
                    epochs=num_epochs,
                    verbose=1
                    )

def plot_graphs(history, string):
    plt.figure()

    plt.plot(history.history[string])           # hisory.history['accuracy']는 30개 짜리 list
    plt.plot(history.history['val_' + string])

    plt.xlabel("Epochs")
    plt.ylabel(string)

    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

# Visualize Word Embedding
reverse_word_index = tokenizer.index_word
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]    # 학습된 embedding weights
print(embedding_weights.shape)

# out_v 는 word 에 따른 embedding 숫자들
# out_m 은 word 들만
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):

    word_name = reverse_word_index[word_num]
    word_embedding = embedding_weights[word_num]

    out_m.write(word_name + '\n')       # string 으로 줘야댐
    out_v.write('\t'.join([str(element) for element in word_embedding]) + '\n')

out_v.close()
out_m.close()









