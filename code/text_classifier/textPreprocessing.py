import csv  
import re
import string
from os import listdir
import pickle

def preprocessText(text):
    if text == None:
        return ""
    else:
        # Remove all unicode characters
        text = (text.encode('ascii', 'ignore')).decode('utf-8')
        # First we lower case the text
        text = text.lower()
        # remove links
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',text)
        #Remove usernames
        text = re.sub('@[^\s]+','', text)
        # replace hashtags by just words
        text = re.sub(r'#([^\s]+)', r'\1', text)
        #correct all multiple white spaces to a single white space
        text = re.sub('[\s]+', ' ', text)
        # Remove all punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Additional clean up : removing words less than 3 chars, and remove space at the beginning and teh end
        text = re.sub(r'\W*\b\w{1,2}\b', '', text)
        text = text.strip()
        
        return text

def findCsvFilenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

train = open('../../processed_data/text.train','w')  
valid = open('../../processed_data/text.valid','w')

# Load labels
train_labels = pickle.load(open('../../processed_data/training_labels.pkl', 'rb'))
valid_labels = pickle.load(open('../../processed_data/testing_labels.pkl', 'rb'))

# Load text
train_text = pickle.load(open('../../processed_data/text_training.pkl', 'rb'))
valid_text = pickle.load(open('../../processed_data/text_testing.pkl', 'rb'))

line = 0
for txt in train_text:
    text = preprocessText(txt[0])
    print(f'__label__{train_labels[line][0].encode("UTF-8")} {text.encode("UTF-8")}', file=train)
    line = line + 1
    
line = 0
for txt in valid_text:    
    text = preprocessText(txt[0])
    print(f'__label__{train_labels[line][0].encode("UTF-8")} {text.encode("UTF-8")}', file=valid)
    line = line + 1

