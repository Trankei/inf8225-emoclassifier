import torch.nn as nn
import torch

def create_emb_layer(pretrained_word_vectors, non_trainable=False):
    num_embeddings, embedding_dim = pretrained_word_vectors.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.tensor(pretrained_word_vectors)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

class TextToVector(nn.Module):
    def __init__(self, pretrained_word_vectors):
        super(TextToVector, self).__init__()
        self.embedding, self.num_embeddings, self.embedding_dim = create_emb_layer(pretrained_word_vectors)

        
    def forward(self, words):
        word_vectors = []
        for word in words:
            word_vector = self.embedding(word)
            word_vectors.append(word_vector.reshape(1, self.embedding_dim))
    
        # Concatenate all word vectors
        words_matrix = torch.cat(word_vectors, dim=0)

        # Aggregation layer with component-wise max as a function
        words_vector = words_matrix.max(dim=0)[0]

        return words_vector
