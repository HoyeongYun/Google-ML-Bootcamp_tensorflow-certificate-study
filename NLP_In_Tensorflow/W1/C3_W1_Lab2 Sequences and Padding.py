from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# pad_sequneces 는 tokenizer.texts_to_sequences로 생성한 sequences 객체를 padding함

corpus = [
    'i love my dog',
    'i love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')         # OOV => Out of vocabulary
tokenizer.fit_on_texts(corpus)
word_dic = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(corpus)

print('Word dic = ', word_dic)
print('Sequences = ', sequences)
# print(type(sequences))  -> list
# print(sequences.shape)


# Padding이 필요한 이유 -> model의 input으로 줄때 shape를 맞춰야하기 때문
padded = pad_sequences(sequences, maxlen=5, padding='pre', truncating='post')
# -> maxlen 이 가장 긴 문장보다 작은경우에는 default로 맨뒤부터 maxlen개수 만큼이 보존된다 (앞이 짤림)
# default 는 pre
print('\nPadded sequences : ')
print(padded)
# print(type(padded))   -> np.ndarray

# 사전에 없는 단어가 등장
test_corpus = [
    'i really love my dog',
    'my dog loves my manatee'
]

# 당연히 이전 문장에 fit 시켜놨던 tokenizer 사용해야됨
test_seq = tokenizer.texts_to_sequences(test_corpus)

# Tokenizer(oov_token) oov token 지정안하면, 처음 보는단어(word dic에 없는단어)가 나왔을 때, sequence에 아무것도 안넣어준다
print('\nWord dic : ', word_dic)
print('\nTest Sequence : ', test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print('\nPadded Test Sequence : \n', padded)