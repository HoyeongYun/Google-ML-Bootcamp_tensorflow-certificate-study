from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['i love my dog', 'i, love my cat']

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sentences = [
    'i love my dog',
    'I,. love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words=1)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

