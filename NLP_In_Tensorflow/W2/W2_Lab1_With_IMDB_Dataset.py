import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

print('info : ', info)
print(f'imdb : {imdb}\n')

for example in imdb['train'].take(2):
    print(type(example))
    print(example)

# print(list(imdb['train'].take(2).as_numpy_iterator())) # tf.data.Dataset 객체를 iterator로 출력해서 보고 싶으면 as_numpy_iterator

train_data, test_data = imdb['train'], imdb['test']         # imdb['train'] 은 PrefetchDataset

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:         #  prefetchdataset->[tuple->(tf.tensor->sentence1, tf.tensor->label1), tf.tensor->(sentence2, label2), ...]
    # s, l 은 tf.tensor
    training_sentences.append(s.numpy().decode('utf8'))     #character 이므로 decode
    training_labels.append(l.numpy())

for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
max_length =120
embedding_dim = 16
trunc_type = 'post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, truncating=trunc_type)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),          # word embedding 공부 다시
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

num_epochs = 10

model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

## visualization Word Embedding
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]        # layer.get_weights()[0] -> weight , get_weights()[1] -> bias
print('embedding_weights.shape : ', embedding_weights.shape)     # vocab_num * dims

reverse_word_index = tokenizer.index_word           # key : value  --   index : word  여기서 index는 1부터 시작

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

print(reverse_word_index[0])
print(reverse_word_index[1])

for word_num in range(1, vocab_size):

    word_name = reverse_word_index[word_num]

    word_embedding = embedding_weights[word_num]

    out_m.write(word_name + '\n')
    out_v.write('\t'.join([str(x) for x in word_embedding]) + '\n')

out_v.close()
out_m.close()

