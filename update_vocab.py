import json
import argparse

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

updparser = argparse.ArgumentParser()
updparser.add_argument("--domain",            default = "Aluneth")
config = updparser.parse_args()

domain_data = load_json("/Users/melkor/Documents/datasets/Aluneth/Knowledge/u1_knowledge.json")

with open("assets/{}_corpus.txt".format(config.domain.lower()),'w') as txt_file:
    for data in domain_data:
        if len(data["statement"]) > 0:txt_file.write(data["statement"] + "\n")