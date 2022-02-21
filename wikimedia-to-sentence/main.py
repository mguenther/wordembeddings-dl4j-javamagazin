from nltk import tokenize
from nltk import download
from os import walk
from os.path import join
from xml.etree import ElementTree as ET


DIRECTORY = '/tmp/data/wikimedia/text/AA'
FILE = '/tmp/data/wikimedia/text/AA/wiki_00'
FILE_OUT = '/tmp/data/wikimedia/text/combined.txt'
download('punkt')


def files_at(directory):
    files = []
    for (path, directories, filenames) in walk(directory):
        files.extend(filenames)
    return [join(directory, f) for f in files]


with open(FILE_OUT, 'w', encoding='utf-8') as file_out:
    for FILE in files_at(DIRECTORY):
        with open(FILE, 'r', encoding='utf-8') as file_in:
            raw_data = file_in.readlines()
            raw_data = '\n'.join(raw_data)
            raw_data = "<docs>" + raw_data + "</docs>"
            root = ET.fromstring(raw_data)
            for doc in root.findall('doc'):
                text = doc.text.strip().replace('\n', '')
                if not text:
                    continue
                for sentence in tokenize.sent_tokenize(text, language='german'):
                    stripped_sentence = sentence.strip()
                    if not stripped_sentence:
                        continue
                    file_out.write(stripped_sentence)
                    file_out.write('\n')
                    print(stripped_sentence)
