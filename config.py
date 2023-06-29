import argparse
from models import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",                 default = device)

# language model configuration
parser.add_argument("--name",                   default = "SkullOfTheManari")
parser.add_argument("--domain",                 default = "aluneth")
parser.add_argument("--corpus_path",            default = "assets/corpus.txt")
parser.add_argument("--num_words",              default = int(1e5))
parser.add_argument("--word_dim",               default = 132)
parser.add_argument("--semantics_dim",          default = 232)

# reasoning model configuration
parser.add_argument("--concept_type",           default = "box")
parser.add_argument("--concept_dim",            default = 100)
parser.add_argument("--object_dim",             default = 100)
parser.add_argument("--temperature",            default = 0.02)

parser.add_argument("--method",                 default = "uniform")
parser.add_argument("--offset",                 default = [-.25, .25])
parser.add_argument("--center",                 default =[.0, .0])
parser.add_argument("--entries",                default = 32)
parser.add_argument("--translator",             default = translator)

config = parser.parse_args(args = [])