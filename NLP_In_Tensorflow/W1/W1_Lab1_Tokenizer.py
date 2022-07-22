from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'i love my dog',
    'i, love my cat'
]

tokenizer = Tokenizer(num_words=100)    # 이 num_words 지정은 word_index에 덜 들어가게 하는 것이 아니라 일단 word_index에는 다 들어가고 sequence에 몇개까지 포함시킬지를 지정해주는 것이다.
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