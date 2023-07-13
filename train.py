from logging import raiseExceptions

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import datetime
import sys
from datasets import *
from models import *
from datasets import *
from config import *
from torch.utils.tensorboard import SummaryWriter

def train_knowledge_prior(model, config, args):
    alpha = 1.0
    beta  = 1.0
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
    translate_program = args.train_translate
    itrs = 1
    for epoch in range(epochs):
        for sample in train_loader:
            inputs = sample

            # [Train Language->Program]
            translate_loss = 0
            if translate_program and len(inputs["program"])!=0 and len(inputs["statement"])!=0:
                translate_loss += 0

            # [Logical Statement Loss]
            statement_loss = 0
            avg_confidence = 0.0
            count_statement = 0
            for b in range(len(inputs["program"])):
                if len(inputs["program"][b])!=0:
                    program = inputs["program"][b]
                    kwargs = {}
                    q = model.executor.parse(program)   
                    o = model.executor(q, **kwargs)
                    statement_loss -= o["end"].squeeze()
                    avg_confidence += torch.sigmoid(o["end"].squeeze())
                    count_statement += 1
            avg_confidence /= count_statement

            working_loss = translate_loss/args.batch_size * alpha + statement_loss * beta

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            
            if not itrs % args.checkpoint_itrs:
                torch.save(model.state_dict(), "checkpoints/alueth.pth")
            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, AvfConi: {}, Time: {}".format(epoch + 1, itrs, working_loss,avg_confidence,datetime.timedelta(seconds=time.time() - start)))
            itrs += 1


def train_neuro_search(model, config, args):
    pass

argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--name",                    default = "LW")
argparser.add_argument("--dataset",                 default = "Aluneth")
argparser.add_argument("--epochs",                  default = 5000)
argparser.add_argument("--batch_size",              default = 2)
argparser.add_argument("--lr",                      default = 2e-3)
argparser.add_argument("--phase",                   default = "knowledge_prior")
argparser.add_argument("--train_translate",         default = True)

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)

# visualization and log contents save iters args
argparser.add_argument("--checkpoint_itrs",         default = 100)

args = argparser.parse_args()

if not args.checkpoint:
    model = MetaReasoner(config)
else:
    model = MetaReasoner(config)
    model.load_state_dict(torch.load(args.checkpoint))

if args.phase == "knowledge_prior":
    train_knowledge_prior(model, config, args)
if args.phase == "neuro_search":
    train_neuro_search(model, config, args)

