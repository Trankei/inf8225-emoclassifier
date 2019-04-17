import bcolz
import pickle
import numpy as np

# Reference : https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'glove.dat', mode='w')

with open(f'glove.twitter.27B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

# Ne fonctionne pas à cause du reshape((nbre de mots dans le vocabulaire, dimension))
vectors = bcolz.carray(vectors[1:].reshape((1193514, 200)), rootdir=f'glove.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'glove_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'glove_idx.pkl', 'wb'))