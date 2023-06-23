import argparse
from models import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

# language model configuration
parser.add_argument("--corpus_path",            default = "assets/corpus.txt")
parser.add_argument("--num_words",              default = int(1e5))
parser.add_argument("--word_dim",               default = 132)
parser.add_argument("--semantics_dim",          default = 232)

# reasoning model configuration
parser.add_argument("--concept_type",           default = "box")
parser.add_argument("--box_dimension",          default = 100)


config = parser.parse_args(args = [])