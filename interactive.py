import torch
import torch.nn as nn

from models import *
from config import *


argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--dataset",                 default = "Aluneth")

argparser.add_argument("--phase",                   default = "knowledge_prior")

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)

args = argparser.parse_args()


if not args.checkpoint:
    model = MetaLearner(config)
else:
    model = MetaLearner(config)
    model.load_state_dict(torch.load(args.checkpoint))