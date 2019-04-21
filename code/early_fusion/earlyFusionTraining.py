from earlyFusionModel import EarlyFusionModel
from textToVector import TextToVector
from textPreprocessing import preprocessText
import torch.nn as nn
import torch.optim as optim
import gloveLoad
import pickle
import torch
import numpy as np

# Image dataset
image_training_features = pickle.load(open('../../processed_data/image_training_features.pkl', 'rb'))
image_validation_features = pickle.load(open('../../processed_data/image_validation_features.pkl', 'rb'))
image_testing_features = pickle.load(open('../../processed_data/image_testing_features.pkl', 'rb'))

# Text dataset
text_training = pickle.load(open('../../processed_data/text_training.pkl', 'rb'))
text_validation = pickle.load(open('../../processed_data/text_validation.pkl', 'rb'))
text_testing = pickle.load(open('../../processed_data/text_testing.pkl', 'rb'))

# Labels
training_labels = pickle.load(open('../../processed_data/training_labels.pkl', 'rb'))
validation_labels = pickle.load(open('../../processed_data/validation_labels.pkl', 'rb'))
testing_labels = pickle.load(open('../../processed_data/testing_labels.pkl', 'rb'))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
one_hot_training_labels = enc.fit_transform(training_labels)

# Number of emotion classes
num_classes = 4

# Image to vector model
# image_to_vect_model = pickle.load(open('../../models/image_classifier_71precision.pkl', 'rb')) #dont need it image already a feature
# Image vector dim
image_vector_dim = 2048

# word_dictionary: Mapping words to index in vocabulary (to convert words to indexes before passing to model)
word_dictionary, pretrained_word_vectors = gloveLoad.create_dictionary(text_training)
text_to_vect_model = TextToVector(pretrained_word_vectors)
text_vector_dim = 200

early_fusion_model = EarlyFusionModel(image_vector_dim, text_to_vect_model, text_vector_dim, num_classes)

def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, label in enumerate(one_hot_training_labels):
            # get the inputs
            image_feature = image_training_features[i]
            tensor_image = torch.LongTensor(image_feature)
            text = text_training[i][0]
            text = preprocessText(text)
            array_text = []
            for word in text.split(): 
                idx = word_dictionary[word]
                array_text.append(idx)
            tensor_text = torch.LongTensor(array_text)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(tensor_image, tensor_text)
            idx = torch.tensor(np.argwhere(label[0] == 1)[0,1])
            loss = criterion(output, idx)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    return model

# def valid(model):
#     # TODO: Implement validation
#     return loss, precision

# def test(model):
#     # TODO: Implement testing
#     return loss, precision

early_fusion_model = train(early_fusion_model)




