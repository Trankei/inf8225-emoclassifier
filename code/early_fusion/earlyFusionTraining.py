from earlyFusionModel import EarlyFusionModel
from textToVector import TextToVector
from textPreprocessing import preprocessText
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch.optim as optim
import gloveLoad
import pickle
import torch
import numpy as np

# Image dataset
image_training_features = pickle.load(open('../../processed_data/image_training_features.pkl', 'rb'))
image_testing_features = pickle.load(open('../../processed_data/image_testing_features.pkl', 'rb'))

# Text dataset
text_training = pickle.load(open('../../processed_data/text_training.pkl', 'rb'))
text_testing = pickle.load(open('../../processed_data/text_testing.pkl', 'rb'))

# Labels
training_labels = pickle.load(open('../../processed_data/training_labels.pkl', 'rb'))
testing_labels = pickle.load(open('../../processed_data/testing_labels.pkl', 'rb'))

# Convert string labels to one hot
enc = OneHotEncoder()
one_hot_training_labels = enc.fit_transform(training_labels)
one_hot_testing_labels = enc.transform(testing_labels)

# Number of emotion classes
num_classes = 4

# Image to vector model
# Images already saved as features, don't need the model
# image_to_vect_model = pickle.load(open('../../models/image_classifier_71precision.pkl', 'rb'))

# Image vector dim, Text vector dim
image_vector_dim = 2048
text_vector_dim = 200

all_text = np.append(text_training, text_testing).reshape((-1, 1))
# word_dictionary: Mapping words to index in vocabulary (to convert words to indexes before passing to model)
word_dictionary, pretrained_word_vectors = gloveLoad.create_dictionary(all_text)
text_to_vect_model = TextToVector(pretrained_word_vectors)

# Get early fusion model
early_fusion_model = EarlyFusionModel(image_vector_dim, text_to_vect_model, text_vector_dim, num_classes)

def train(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, label in enumerate(one_hot_training_labels):
            
            # Get the inputs : image
            image_feature = image_training_features[i]
            tensor_image = torch.FloatTensor(image_feature)
            
            # Get the inputs : text
            text = text_training[i][0]
            text = preprocessText(text)
            array_text = []
            for word in text.split(): 
                idx = word_dictionary[word]
                array_text.append(idx)
            tensor_text = torch.LongTensor(array_text)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + Backward + Optimize
            output = model(tensor_image, tensor_text)
            output = output.reshape((-1,4))
            
            target = torch.tensor(np.argwhere(label[0] == 1)[0,1])
            target = target.reshape((-1,))
            target = target.type(torch.LongTensor)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:    # print loss every 400 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
    return model

def test(model):
    correct = 0
    total = 0
    targets = []
    predictions = []
    criterion_test = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, test_label in enumerate(one_hot_testing_labels):
            # Get the inputs : image
            image_feature = image_testing_features[i]
            tensor_image = torch.FloatTensor(image_feature)
            
            # Get the inputs : text
            text = text_testing[i][0]
            text = preprocessText(text)
            array_text = []
            for word in text.split():
                try:
                    idx = word_dictionary[word]
                except KeyError:
                    print('Word not in vocabulary')
                else:
                    array_text.append(idx)
            if len(array_text) > 0:
                tensor_text = torch.LongTensor(array_text)

                # Get model prediction
                output = model(tensor_image, tensor_text)
                output = output.reshape((-1,4))
                _, predicted = torch.max(output.data, 1)
                predictions.append(output.numpy().squeeze())
                
                # Get target index
                idx_label = np.argwhere(test_label[0] == 1)[0,1]
                targets.append(idx_label)
                
                # Count correct answers
                correct += (predicted == idx_label).sum().item()
                total += 1
     
    # Get precision and loss
    precision = (100 * correct / total)
    targets = torch.LongTensor(targets)
    predictions = torch.FloatTensor(predictions)
    loss = criterion_test(predictions, targets)
    return loss, precision

# early_fusion_model = torch.load(f'../../models/early_fusion_model.tar')
early_fusion_model = train(early_fusion_model)
loss, precision = test(early_fusion_model)
torch.save(early_fusion_model, f'../../models/early_fusion_model.tar')
print('Accuracy of the network on the 639 test examples: %d %%' % (precision))
print('Loss of the network on the 639 test examples: %.3f ' % (loss))




