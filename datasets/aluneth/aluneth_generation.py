import numpy as np
import json

"""
[Knowledge]
for a data bind of knowledge base
1. Natural Language (Theorem) Statement.
2. LEAN statement.
3. Aluneth Executable Program.
"""

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

data = [
    {"statement":"A Banach space is a complete normed space",
    "lean":[],
    "program":"subset(banach_space,cintersect(complete,normed))"},

    {"statement":"A Hilbert space is a Banach space with inner product",
    "lean":[],
    "program":"subset(hilbert_space,cintersect(banach_space,has_inner_product))"},

    {"statement":"A compact set must be closed and bounded.",
    "lean":[],
    "program":"subset(compact,cintersect(close,bounded))"},

    {"statement":"A function is a map.",
    "lean":[],
    "program":"subset(function,map)"},

]

if __name__ == "__main__":
    save_json(data,"/Users/melkor/Documents/datasets/Aluneth/Knowledge/u1_knowledge.json")

    states = load_json("/Users/melkor/Documents/datasets/Aluneth/Knowledge/u1_knowledge.json")
