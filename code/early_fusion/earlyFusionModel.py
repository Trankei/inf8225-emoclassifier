import torch
import torch.nn as nn
import torch.nn.functional as F
from textToVector import TextToVector

class EarlyFusionModel(nn.Module):
    def __init__(self, image_to_vect_model, image_vect_dim, text_to_vect_model, text_vect_dim, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.image_to_vect = image_to_vect_model
        self.text_to_vect = text_to_vect_model
        self.linear_layer = nn.Linear(image_vect_dim + text_vect_dim, num_classes)
        self.softmax_layer = nn.Softmax()
        
    def forward(self, image, text):
        # Convert image to vector
        # Modify code if not compatible with Keras image to vector model
        image_vect = self.image_to_vect(image)
        # Convert text to vector
        text_vect = self.text_to_vect(text)

        # Concatenation of image and text vector (Fusion layer)
        concat_vect = torch.cat((image_vect, text_vect), dim=0)
        # Pass through linear layer
        output = self.linear_layer(concat_vect)
        # Softmax
        output = self.softmax_layer(output)
        return output
