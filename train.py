from logging import raiseExceptions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from datasets import *
from visualization import *
from models import *
from datasets import *
from config import *
from torch.utils.tensorboard import SummaryWriter

def train_knowledge_prior(model, config, args):
    epochs = args.epochs
    if args.dataset == "Aluneth":
        if args.phase == "knowledge_prior":
            train_dataset = AlunethKnowledge(config)
        elif args.phase == "neuro_search":
            train_dataset = AlunethSearch(config)
        else:
            raiseExceptions
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)
    
    # [Initalize the model optimizer]
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)


    itrs = 0
    for epoch in range(epochs):
        for sample in train_loader:
            inputs = sample

            outputs = model(inputs)

            # [Logical Statement Loss]
            statement_loss = 0
            working_loss = statement_loss * 1.0

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            if not itrs % args.checkpoint_itrs:
                torch.save(model.state_dict(), "alueth.pth")


def train_neuro_search(model, config, args):
    pass

argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--name",                    default = "LW")
argparser.add_argument("--dataset",                 default = "Aluneth")
argparser.add_argument("--epochs",                  default = 1000)
argparser.add_argument("--batch_size",              default = 2)
argparser.add_argument("--lr",                      default = 2e-4)
argparser.add_argument("--phase",                   default = "knowledge_prior")

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)

# visualization and log contents save iters args
argparser.add_argument("--checkpoint_itrs",         default = 100)

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