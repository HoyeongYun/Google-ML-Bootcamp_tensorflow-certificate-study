import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

imdb_plaintext, info_plaintext = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
imdb_subwords, info_subwords = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

print(info_plaintext.features)

for example in imdb_plaintext['train'].take(2):
    print(example)
# imdb_plaintext는 sequence가 character로 이루어져있지만, imdb_subwords는 int64 로 이루어져있다
print('info_subwords.features : ', info_subwords.features)

# for example in imdb_subwords['train'].take(2):
#     print(example)
#
tokenizer_subwords = info_subwords.features['text'].encoder
#
# for example in imdb_subwords['train'].take(2):
#     print(tokenizer_subwords.decode(example[0]))
#     print(example[0])

train_data = imdb_plaintext['train']
training_sentences = []

for s, _ in train_data:

    # print(s) # tf.Tensor(b"~~", shape=(), dtype=string)
    # print(s.numpy())    # b"~~"
    training_sentences.append(s.numpy().decode('utf8'))     # "~~"

vocab_size = 10000
oov_tok = "<OOV>"

tokenizer_plaintext = Tokenizer(num_words=10000, oov_token=oov_tok)
tokenizer_plaintext.fit_on_texts(training_sentences)        # fit_on_texts 만 해도 word_index가 생성된다
sequences = tokenizer_plaintext.texts_to_sequences(training_sentences)

# print(sequences)   # [[1, 23, 25, ...], [], [], ...]


# OOV 문제
# training 문장에 대해서 sequence로 만든걸 고대로 다시 text로 했는데 원래 text 복원이 안돼서(OOV 등장) 이상하다 생각했는데, 이유는 num_words보다 많은 단어가 training_sentences에 있어서(88583개) 그중 만개 빼고는 다 OOV 처리됨
# 기계가 문제를 풀 때, 모르는 단어가 등장하면 (사람도 마찬가지지만) 주어진 문제를 푸는 것이 까다로워 집니다. 이와 같이 모르는 단어로 인해 문제를 푸는 것이 까다로워지는 상황을 OOV 문제라고 합니다.
# 서브워드 분리작업은 하나의 단어는 더 작은 단위의 의미있는 여러 서브워드들(Ex) birthplace = birth + place)의 조합으로 구성된 경우가 많기 때문에,
# 하나의 단어를 여러 서브워드로 분리해서 단어를 인코딩 및 임베딩하겠다는 의도를 가진 전처리 작업입니다. 이를 통해 희귀단어나 신조어와 같은 문제를 완화시킬 수 있다
print(tokenizer_plaintext.sequences_to_texts(sequences[0:1]))   # sequences_to_texts( iterable )  iterable을 줘야해서 sequences[0]으로 주면 안됨

print(len(tokenizer_plaintext.word_index))          # 88583

# 위에서 등장한 문제 -> 88583 개의 단어중 10000개만 쓰자니 oov 처리되는게 싫고 다쓰자니 model이 bloat 되고 학습시간도 slow down 된다

print(tokenizer_subwords.subwords)                  # 이게 뭐야  tokenizer_subwords = info_subwords.features['text'].encoder
# #['the_', ', ', '. ', 'a_', 'and_', 'of_', 'to_', 's_', 'is_', 'br', 'in_', 'I_', 'that_', 'this_', 'it_', ' /><', ' />', 'w ...]
# 이걸 사용하면 위 문제를 해결해준다는데, 어떻게 한다는 거야

tokenized_string = tokenizer_subwords.encode(training_sentences[0])
print("training 의 첫번째 sentence를 subwords encoder로 인코딩 한 결과 : ", tokenized_string)

original_string = tokenizer_subwords.decode(tokenized_string)
print("위에 인코딩한걸 그대로 다시 디코딩한거 : ", original_string)
## https://wikidocs.net/22592 이거 읽고 이해


# movie review 아닌 텍스트에 해도 subword tokenize 하면 원래 문장 잘 되돌릴 수 있다고 한다. 어차피 단어가 합성어가 많고 재사용 되는 subword 들이 많기 때문에
sample_string = 'TensorFlow, from basics to mastery'

# plaintext Tokenizer 사용 case
tokenized_string = tokenizer_plaintext.texts_to_sequences([sample_string])
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_plaintext.sequences_to_texts(tokenized_string)
print('The original string : {}'.format(original_string))

# subword Tokenizer 사용 case
tokenized_string = tokenizer_subwords.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_subwords.decode(tokenized_string)
print('The original string: {}'.format(original_string))

for ts in tokenized_string:
    print('{} ---> {}'.format(ts, tokenizer_subwords.decode([ts])))


# train model
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_data, test_data = imdb_subwords['train'], imdb_subwords['test']

train_dataset = train_data.shuffle(BUFFER_SIZE)

train_dataset = train_dataset.padded_batch(BATCH_SIZE)
test_dataset = test_data.padded_batch(BATCH_SIZE)

embedding_dim = 64

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer_subwords.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

num_epochs = 10

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_dataset,
                    epochs=num_epochs,
                    validation_data=test_dataset)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])

    plt.xlabel('Epochs')
    plt.ylabel(string)

    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')