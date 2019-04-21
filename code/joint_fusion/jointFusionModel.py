import torch
import torch.nn as nn
import torch.nn.functional as F
from textToVector import TextToVector

class JointFusionModel(nn.Module):
    def __init__(self, image_vect_dim, text_to_vect_model, text_vect_dim, num_classes):
        super(JointFusionModel, self).__init__()
        self.text_to_vect = text_to_vect_model
        self.image_features_projection = nn.Linear(image_vect_dim, text_vect_dim)
        self.linear_layer = nn.Linear(text_vect_dim, num_classes)
        self.softmax_layer = nn.Softmax()
        
    def forward(self, image_vect, text):
        # Convert text to vector
        # Note: input text should be a tensor of integers obtained from word dictionary lookup (see word_dictionary in earlyFusionTraining.py)
        text_vect = self.text_to_vect(text)
        text_vect = text_vect.type(torch.LongTensor)

        # Linear projection of image features into textual feature space
        image_vect = image_vect.type(torch.FloatTensor)
        new_image_vect = self.image_features_projection(image_vect)
        
        # Max pooling of the two vectors
        max_pooling_result = torch.max(new_image_vect.type(torch.FloatTensor), text_vect.type(torch.FloatTensor))
        
        # Linear layer 
        output = self.linear_layer(max_pooling_result)
        
        # Softmax
        output = self.softmax_layer(output)
        return output
