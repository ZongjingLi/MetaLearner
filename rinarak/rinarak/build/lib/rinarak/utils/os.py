import os
import json
import matplotlib.pyplot as plt

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