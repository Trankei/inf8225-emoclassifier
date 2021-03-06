import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from textToVector import TextToVector

class EarlyFusionModel(nn.Module):
    def __init__(self, image_vect_dim, text_to_vect_model, text_vect_dim, num_classes):
        super(EarlyFusionModel, self).__init__()
        self.text_to_vect = text_to_vect_model
        self.linear_layer = nn.Linear(image_vect_dim + text_vect_dim, 100)
        self.hidden_layer = nn.Linear(100, num_classes)
        self.softmax_layer = nn.Softmax()
        
    def forward(self, image_vect, text):
        # Convert text to vector
        # Note: input text should be a tensor of integers obtained from word dictionary lookup (see word_dictionary in earlyFusionTraining.py)
        text_vect = self.text_to_vect(text)

        # Concatenation of image and text vector (Fusion layer)
        concat_vect = torch.cat((image_vect, text_vect), dim=0)
        # Pass through linear layer
        output = self.linear_layer(concat_vect)
        #Pass through hidden layer
        output = self.hidden_layer(output)
        # Softmax
        output = self.softmax_layer(output)
        return output
