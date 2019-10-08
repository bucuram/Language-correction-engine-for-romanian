import numpy as np
import re

characters = '-.,!?aăâbcdefghiîjklmnopqrsștțuvwxyzAĂÂBCDEFGHIÎJKLMNOPQRSȘTȚUVWXYZ '

def clean(file):
    text = open(file).read().splitlines()
    text = [re.sub(r'\s([?.!,"](?:\s|$))', r'\1', re.sub(r'(\[[^)]*\])|([^' + characters + '])|(?:(?<=-) | (?=-))', '', line)) for line in text]
    # text = list(filter(None, text))
    # text = [line for line in text if len(line)>=25 and len(line)<=100]

    # print((text[:20]))

    # text = [re.sub(' +', ' ', line) for line in text]
    # text = [re.sub(r'\.{2,}', '.', line) for line in text]
    text = [line for line in text if 'px' not in line]

    with open('clean_text6.txt', 'w') as out_file:
        out_file.write('\n'.join(text))
    print(len(text))
file_path = './clean_text5.txt'
clean(file_path)
