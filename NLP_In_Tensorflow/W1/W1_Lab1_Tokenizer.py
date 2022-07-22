from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'i, love my cat'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sentences = [
    'i love my dog',
    'I,. love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

sentences = [
    'i, a b ,ccd efa',
    'ge dfe',
    'i i i i a a b b ccd ccd'
]

tokenizer = Tokenizer(num_words=3)
tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)