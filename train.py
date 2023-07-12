from socket import AI_PASSIVE
import torch
import torch.nn as nn

from visualization import *

from models import *
from datasets import *

from config import *

def train_knowledge_prior(model, config, args):
    pass

def train_neuro_search(model, config, args):
    pass

argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--epoch",                   default = 1000)
argparser.add_argument("--batch_size",              default = 2)
argparser.add_argument("--lr",                      default = 2e-4)
argparser.add_argument("--phase",                   default = "knowledge_prior")

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)
args = argparser.parse_args()

if not args.checkpoint:
    model = MetaReasoner(config)
else:
    model = MetaReasoner(config)
    model.load_state_dict(args.checkpoint)

if args.phase == "knowledge_prior":
    train_knowledge_prior(model, config, args)
if args.phase == "neuro_search":
    train_neuro_search(model, config, args)

concepts = ["topological_vector_space",
            "banach_steinhaus_theorem",
            "close", "open", "compact",
            "seperation", "banach_space",
            "hilbert_space", "heine_borel",
            "bounded", "continuous", "function",
            "map", "mapping","manifold"]

visualize_concepts(concepts,model)
plt.show()