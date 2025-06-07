import os
import json
#import matplotlib.pyplot as plt
from typing import List

def load_json(path):
    with open(path,'r') as f:
        data = json.load(f)
        return data

def save_json(data,path):
    '''input the diction data and save it'''
    beta_file = json.dumps(data)
    file = open(path,'w')
    file.write(beta_file)
    return True

def save_im(im_arr, path = "tmp.png"):
    plt.imsave(path,im_arr)


def load_corpus(corpus_path : str) -> List[str]:    
    sequences = []
    with open(corpus_path) as corpus:
        for line in corpus:
            line = line.strip()
            if line:
                line = line.lower()
                line = ' '.join(line.split())
                sequences.append(line)
    return sequences

