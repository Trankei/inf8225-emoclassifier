import torch
import torch.nn as nn
import torch.nn.functional as F
from textToVector import TextToVector

class JointFusionModel(nn.Module):
    def __init__(self, image_vect_dim, text_to_vect_model, text_vect_dim, num_classes):
        super(JointFusionModel, self).__init__()
        self.text_to_vect = text_to_vect_model
        self.image_features_projection = nn.Linear(image_vect_dim, text_vect_dim)
        self.linear_layer = nn.Linear(text_vect_dim, 100)
        self.hidden_layer = nn.Linear(100, num_classes)
        self.softmax_layer = nn.Softmax()
        
    def forward(self, image_vect, text):
        text_exists = False
        image_exists = False
        
        # Convert text to vector
        # Note: input text should be a tensor of integers obtained from word dictionary lookup (see word_dictionary in earlyFusionTraining.py)
        if text is not None:
            text_vect = self.text_to_vect(text)
            text_exists = True
        
        # Linear projection of image features into textual feature space
        if image_vect is not None:
            image_vect = image_vect.type(torch.FloatTensor)
            new_image_vect = self.image_features_projection(image_vect)
            image_exists = True
        
        # Max pooling of the two vectors
        if text_exists and not image_exists:
            max_pooling_result = text_vect
        elif not text_exists and image_exists:
            max_pooling_result = new_image_vect
        else:
            max_pooling_result = torch.max(new_image_vect, text_vect)
        
        # Linear layer 
        output = self.linear_layer(max_pooling_result)
        
        # Hidden layer
        output = self.hidden_layer(output)

        # Softmax
        output = self.softmax_layer(output)

        return output
