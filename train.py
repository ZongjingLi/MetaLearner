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

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def train_knowledge_prior(model, config, args):
    alpha = args.alpha
    beta  = args.beta
    EPS = 1e-6
    epochs = args.epochs
    if args.dataset == "Aluneth":
        if args.phase in ["knowledge_prior", "translation"]:
            train_dataset = AlunethKnowledge(config)
        elif args.phase == "neuro_search":
            train_dataset = AlunethSearch(config)
        else:
            raiseExceptions
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)
    
    if args.freeze_concepts:freeze_parameters(model.executor)
        
    
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
            avg_translation_conf = 0.0
            count_trans = 0
            if args.phase in ["translation"]:
                if translate_program and len(inputs["program"])!=0 and len(inputs["statement"])!=0:
                    outputs = model.translate(inputs["statement"], inputs["program"])
                    for term in outputs["loss"]:
                        translate_loss += term
                        count_trans += 1
                        avg_translation_conf += torch.exp(0 - term)

                avg_translation_conf /= count_trans
                writer.add_scalar("translate_loss",translate_loss.detach(),itrs)
                writer.add_scalar("avg_translate_conf",avg_translation_conf,itrs)
            
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
                    statement_loss -= o["end"].squeeze(0)
                    avg_confidence += torch.sigmoid(o["end"].squeeze())
                    count_statement += 1
            avg_confidence /= count_statement

            
            working_loss = statement_loss * alpha + translate_loss * beta
            writer.add_scalar("statement_loss",statement_loss.detach(),itrs)
            writer.add_scalar("avg_statement_conf",avg_confidence,itrs)

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            
            if not itrs % args.checkpoint_itrs:
                torch.save(model.state_dict(), "checkpoints/alueth.pth")
            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, AvgConf: {} AvgTrans: {} , Time: {}".format(epoch + 1, itrs, working_loss,avg_confidence,avg_translation_conf,datetime.timedelta(seconds=time.time() - start)))
            itrs += 1


def train_neuro_search(model, config, args):
    pass

argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--name",                    default = "LW")
argparser.add_argument("--dataset",                 default = "Aluneth")
argparser.add_argument("--epochs",                  default = 5000)
argparser.add_argument("--batch_size",              default = 5)
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--phase",                   default = "knowledge_prior")
argparser.add_argument("--train_translate",         default = True)
argparser.add_argument("--freeze_concepts",         default = False)

argparser.add_argument("--alpha",                   default = 1.0)
argparser.add_argument("--beta",                    default = 1.0)

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)

# visualization and log contents save iters args
argparser.add_argument("--checkpoint_itrs",         default = 100)

args = argparser.parse_args()
args.lr = float(args.lr)

if not args.checkpoint:
    model = MetaLearner(config)
else:
    model = MetaLearner(config)
    model.load_state_dict(torch.load(args.checkpoint))

if args.phase in  ["knowledge_prior","translation" ]:
    train_knowledge_prior(model, config, args)
if args.phase == "neuro_search":
    train_neuro_search(model, config, args)

