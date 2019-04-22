import bcolz
import pickle
import numpy as np
import os
from textPreprocessing import preprocessText

glove_word_vectors_path = f'../../word_vectors/glove.twitter.27B.200d.txt'
glove_data_path = f'../../word_vectors/glove.dat'
glove_words_path = f'../../word_vectors/glove_words.pkl'
glove_idx_path = f'../../word_vectors/glove_idx.pkl'

# Reference : https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
def parse_glove():
    words = []
    idx = 0
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=glove_data_path, mode='w')

    with open(glove_word_vectors_path, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    vectors = bcolz.carray(vectors[0:].reshape((-1, 200)), rootdir=glove_data_path, mode='w')
    vectors.flush()
    pickle.dump(words, open(glove_words_path, 'wb'))
    pickle.dump(word2idx, open(glove_idx_path, 'wb'))

def create_dictionary(text_training):
    if not os.path.isdir(glove_data_path) or not os.path.isfile(glove_words_path) or not os.path.isfile(glove_idx_path):
        print('Glove data not found. Parsing data...')
        parse_glove()
        print('Finished parsing glove data.')

    vectors = bcolz.open(glove_data_path)[:]
    words = pickle.load(open(glove_words_path, 'rb'))
    word2idx = pickle.load(open(glove_idx_path, 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    vocab = []
    dictionary = {}
    num_words = 0
    for i, sentence in enumerate(text_training):
        sentence = preprocessText(sentence[0])
        split_sent = sentence.split()
        for j, word in enumerate(split_sent):
            if word not in vocab:
                vocab.append(word)
                dictionary[word] = num_words
                num_words += 1
    vocab = np.array(vocab)

    word_vector_dim = vectors.shape[1]
    matrix_len = len(vocab)
    word_vectors = np.zeros((matrix_len, word_vector_dim))
    words_found = 0

    for i, word in enumerate(vocab):
        try: 
            word_vectors[i] = glove[word]
            words_found += 1
        except KeyError:
            word_vectors[i] = np.random.normal(scale=0.6, size=(word_vector_dim, ))

    print(f'{words_found} word vocabulary created')

    return dictionary, word_vectors