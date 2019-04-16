import csv
import requests
from bs4 import BeautifulSoup
from os import listdir
import threading

def extractFlickrImageId(image_url):
    nFrontSlash = 0
    for i, c in enumerate(image_url):
        if c == '/':
            nFrontSlash = nFrontSlash + 1
            if nFrontSlash == 4:
                start_i = i + 1
        if c == '_':
            end_i = i
            break

    return image_url[start_i:end_i]


def crawlTitleAndDesc(image_url):
    image_id = extractFlickrImageId(image_url)
    url = "https://flickr.com/photo.gne?id=" + image_id
    try:
        request = requests.get(url)
    except:
        print("Request error")
        return None, None
        
    html = request.content
    soup = BeautifulSoup(html, features="lxml")

    title_meta = soup.find('meta', property="og:title")
    if title_meta != None:
        title = title_meta["content"]
    else:
        title = ""

    description_meta = soup.find('meta', property="og:description")
    if description_meta != None:
        description = description_meta["content"]
    else:
        description = ""

    # Remove all unicode characters
    title = (title.encode('ascii', 'ignore')).decode('utf-8')
    description = (description.encode('ascii', 'ignore')).decode('utf-8')

    return title, description

def findCsvFilenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def flicklrCrawler(emotion):
    dataPath = '../../data/flickr_data/' + emotion + '/'
    filenames = findCsvFilenames(dataPath)
    for name in filenames:
        if name.find('ins') == -1:
            ext_index = name.find('.csv')
            processed_name = name[:ext_index] + "_processed" + name[ext_index:]
            with open(dataPath + name, mode='r', encoding = "ISO-8859-1") as input_csv:
                with open('../../data/flickr_data/processed/' + processed_name, mode='a', encoding = "ISO-8859-1") as output_csv:
                    reader = csv.reader(input_csv)
                    writer = csv.writer(output_csv, lineterminator='\n')
                    i = 0
                    for row in reader:
                        if (i % 100 == 0):
                            print(f'[{emotion}] Rows processed: {i}')
                        if row[3] > row[2]:
                            title, description = crawlTitleAndDesc(row[1])
                            if (title != None and description != None):
                                row.append(f'{title} {description}')
                                writer.writerow(row)
                        i = i + 1
                output_csv.close()
            input_csv.close()
    return

flickrDatasetEmotions = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

thread_list = []
for emotion in flickrDatasetEmotions:
    thread = threading.Thread(target=flicklrCrawler, args=(emotion,))
    thread_list.append(thread)
    thread.start()

