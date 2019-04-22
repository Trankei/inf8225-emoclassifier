from jointFusionModel import JointFusionModel
from textToVector import TextToVector
from textPreprocessing import preprocessText
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
import torch.optim as optim
import gloveLoad
import pickle
import torch
import numpy as np

from sklearn.model_selection import train_test_split

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

# Split test into test and validation 
image_features_test, image_features_valid, one_hot_labels_test, one_hot_labels_valid = train_test_split(image_testing_features, one_hot_testing_labels, test_size=0.5, random_state=42)
text_test, text_valid, one_hot_labels_test, one_hot_labels_valid = train_test_split(text_testing, one_hot_testing_labels, test_size=0.5, random_state=42)


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
joint_fusion_model = JointFusionModel(image_vector_dim, text_to_vect_model, text_vector_dim, num_classes)

def test(model, image_dataset, text_dataset, one_hot_labels_dataset):
    model.eval()
    correct = 0
    total = 0
    targets = []
    predictions = []
    criterion_test = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, test_label in enumerate(one_hot_labels_dataset):
            # Get the inputs : image
            image_feature = image_dataset[i]
            tensor_image = torch.FloatTensor(image_feature)
            
            # Get the inputs : text
            text = text_dataset[i][0]
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

def train(model):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    best_precision = 0
    cnt = 0
    for epoch in range(20):  # loop over the dataset multiple times
        model.train()
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
        
        model.eval()
        loss, precision = test(model, image_features_valid, text_valid, one_hot_labels_valid) 
        if precision > best_precision :
            best_precision = precision
            best_model = type(model)(image_vector_dim, text_to_vect_model, text_vector_dim, num_classes)
            best_model.load_state_dict(model.state_dict())
            cnt = 0
            print('best precision')
            print(best_precision)
        else :
            print('step')
            lr_scheduler.step()
            cnt = cnt + 1
        if cnt > 3 :
            break
    return best_model


def test_only_text(model, text_dataset, one_hot_labels_dataset):
    correct = 0
    total = 0
    targets = []
    predictions = []
    criterion_test = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, test_label in enumerate(one_hot_labels_dataset):
            # Get the inputs : image
            # image_feature = np.full((1, 2048), 0)
            # tensor_image = torch.FloatTensor(image_feature)
            
            # Get the inputs : text
            text = text_dataset[i][0]
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
                output = model(None, tensor_text)
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


def test_only_image(model, image_dataset, one_hot_labels_dataset):
    correct = 0
    total = 0
    targets = []
    predictions = []
    criterion_test = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, test_label in enumerate(one_hot_labels_dataset):
            # Get the inputs : image
            image_feature = image_dataset[i]
            tensor_image = torch.FloatTensor(image_feature)
            
            # Get model prediction
            output = model(tensor_image, None)
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
    precision = (100. * correct / total)
    targets = torch.LongTensor(targets)
    predictions = torch.FloatTensor(predictions)
    loss = criterion_test(predictions, targets)
    return loss, precision

# joint_fusion_model = torch.load('../../models/join_fusion_model.bin')
#joint_fusion_model = train(joint_fusion_model)
#loss, precision = test(joint_fusion_model, image_features_test, text_test, one_hot_labels_test)  
#torch.save(joint_fusion_model, '../../models/join_fusion_model.bin')

joint_fusion_model = torch.load('../../models/join_fusion_model.bin')
print('Accuracy of the network with two modalities: %.2f %%' % (precision))
print('Loss of the network with two modalities: %.3f ' % (loss))
loss, precision = test_only_text(joint_fusion_model, text_test, one_hot_labels_test)
print('Accuracy of the network with text only: %.2f %%' % (precision))
print('Loss of the network with text only: %.3f ' % (loss))
loss, precision = test_only_image(joint_fusion_model, image_features_test, one_hot_labels_test)
print('Accuracy of the network with image only: %.2f %%' % (precision))
print('Loss of the network with image only: %.3f ' % (loss))

