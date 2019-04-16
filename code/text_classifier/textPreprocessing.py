import csv  
import re
import string
from os import listdir

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

train = open('text.train','w')  
test = open('text.valid','w')

redditDatasetFiles = ["processed_creepy.csv", "processed_gore.csv", "processed_happy.csv", "processed_rage.csv"]
for datasetFile in redditDatasetFiles:
    with open("./data/reddit_data/" + datasetFile, mode='r', encoding = "ISO-8859-1") as csv_file:  
        csv_reader = csv.DictReader(csv_file)
        line = 0
        for row in csv_reader:
            line = line + 1
            if line > 0:
                text = preprocessText(row["title"])
                # Split data into train and validation
                if line % 16 == 0:
                    print(f'__label__{row["subreddit"].encode("UTF-8")} {text.encode("UTF-8")}', file=test)
                else:
                    print(f'__label__{row["subreddit"].encode("UTF-8")} {text.encode("UTF-8")}', file=train)

flickrDatasetFiles = findCsvFilenames("./data/flickr_data/crawled")
for datasetFile in flickrDatasetFiles:
    with open("./data/flickr_data/crawled/" + datasetFile, mode='r', encoding = "ISO-8859-1") as csv_file:  
        csv_reader = csv.DictReader(csv_file, fieldnames=["emotion", "image", "num_of_disagrees", "num_of_agrees", "text"])
        line = 0
        for row in csv_reader:
            line = line + 1
            if line > 0:
                text = preprocessText(row["text"])
                if text != "":
                    # Split data into train and validation
                    if line % 16 == 0:
                        print(f'__label__{row["emotion"].encode("UTF-8")} {text.encode("UTF-8")}', file=test)
                    else:
                        print(f'__label__{row["emotion"].encode("UTF-8")} {text.encode("UTF-8")}', file=train)

