import re
import string

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
        #text = re.sub(r'\W*\b\w{1,2}\b', '', text)
        text = text.strip()
    
    return text