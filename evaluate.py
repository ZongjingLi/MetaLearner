from itertools import count
import torch
import torch.nn as nn
from tqdm import tqdm

from datasets import *
from models import *
from config import *

def evaluate_knowledge_prior(model, config, args):
    if args.dataset == "Aluneth":
        if args.phase in ["knowledge_prior", "translation"]:
            train_dataset = AlunethKnowledge(config)
        elif args.phase == "neuro_search":
            train_dataset = AlunethSearch(config)
        else:
            raiseExceptions
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)

    accept_prob = 0.9
    correct_num = 0
    statement_loss = 0
    avg_confidence = 0.0
    count_statement = 0
    
    for inputs in tqdm(train_loader):
        # [Logical Statement Loss]
        for b in range(len(inputs["program"])):
            if len(inputs["program"][b])!=0:
                program = inputs["program"][b]
                kwargs = {}
                q = model.executor.parse(program)   
                o = model.executor(q, **kwargs)
                statement_loss -= o["end"].squeeze(0)
                avg_confidence += torch.sigmoid(o["end"].squeeze())
                count_statement += 1

                if torch.sigmoid(o["end"].squeeze()) > accept_prob:
                    correct_num += 1

    avg_confidence /= count_statement
    print("Average Knowledge Confidence:",avg_confidence.detach().numpy(), "Acc: {}/{}".format(correct_num,count_statement))
    with open("outputs/{}_{}_evaluation.txt".format(args.dataset, args.phase),'w') as evaluation:
        evaluation.write(str("Average Knowledge Confidence:{} Acc: {}/{}".format(avg_confidence.detach().numpy(),correct_num,count_statement)))


def evaluate_translation(model, config, args):
    if args.dataset == "Aluneth":
        if args.phase in ["knowledge_prior", "translation"]:
            train_dataset = AlunethKnowledge(config)
        elif args.phase == "neuro_search":
            train_dataset = AlunethSearch(config)
        else:
            raiseExceptions
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size)

    correct_num = 0
    total_num = 0
    translate_loss = 0
    avg_translation_conf = 0.0
    count_trans = 0
    for inputs in tqdm(train_loader):
        # [Train Language->Program]
        if args.phase in ["translation"]:
            if True and len(inputs["program"])!=0 and len(inputs["statement"])!=0:
                outputs = model.translate(inputs["statement"], )
                for i,term in enumerate(outputs["loss"]):
                    translate_loss += term
                    count_trans += 1
                    avg_translation_conf += torch.exp(0 - term)
                    if str(model.executor.parse(inputs["program"][i])).replace(" ","")\
                        ==str(outputs["program"][i]).replace(" ",""):correct_num += 1
                    total_num += 1
    avg_translation_conf /= count_trans
    print("Average Translation Confidence:",avg_translation_conf.detach().numpy(),"Acc:{}/{}".format(correct_num,total_num))
    with open("outputs/{}_{}_evaluation.txt".format(args.dataset, args.phase),'w') as evaluation:
        evaluation.write(str("Average Translation Confidence:{} Acc: {}/{}".format(avg_translation_conf.detach().numpy(),correct_num,total_num)))


def evaluate_neuro_search(model, config, args):
    pass
argparser = argparse.ArgumentParser()
# basic usage of training session and phase change
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--dataset",                 default = "Aluneth")
argparser.add_argument("--batch_size",              default = 2)
argparser.add_argument("--phase",                   default = "knowledge_prior")

# check for any checkpoints to load
argparser.add_argument("--checkpoint",              default = False)

args = argparser.parse_args()


if not args.checkpoint:
    model = MetaLearner(config)
else:
    model = MetaLearner(config)
    model.load_state_dict(torch.load(args.checkpoint))

if args.phase in  ["knowledge_prior",]:
    evaluate_knowledge_prior(model, config, args)
if args.phase in ["translation" ]:
    evaluate_translation(model, config, args)
if args.phase == "neuro_search":
    evaluate_neuro_search(model, config, args)