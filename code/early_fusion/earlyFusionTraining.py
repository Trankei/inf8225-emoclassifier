from earlyFusionModel import EarlyFusionModel
from textToVector import TextToVector
import gloveLoad
import pickle

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

# Number of emotion classes
num_classes = 4

# TODO: Plug image to vector model
image_to_vect_model = None
# TODO: Plug image vector dim
image_vector_dim = 2048

# word_dictionary: Mapping words to index in vocabulary (to convert words to indexes before passing to model)
word_dictionary, pretrained_word_vectors = gloveLoad.create_dictionary(text_training)
text_to_vect_model = TextToVector(pretrained_word_vectors)
text_vector_dim = 200

early_fusion_model = EarlyFusionModel(image_to_vect_model, image_vector_dim, text_to_vect_model, text_vector_dim, num_classes)

# def train(model):
#     # TODO: Implement training
#     return model

# def valid(model):
#     # TODO: Implement validation
#     return loss, precision

# def test(model):
#     # TODO: Implement testing
#     return loss, precision






