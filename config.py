'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-17 09:41:50
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-17 09:42:32
 # @ Description: This file is distributed under the MIT license.
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--command",          default = "train",   help = "the commands to run for the scripts")
"""possible command types are in the following format
--train_ccsp_{domain} : to train a energy graph diffusion model that learns to do the CCSP problem.
--train_metaphor_{domain} : to train a specific domain information, this domain is supposed to be in the target model.
--learn_new_domain
"""
#TODO: remove this, we don't actually need that curriculum stuff.

parser.add_argument("--generic_dim",      default = 256,       help = "the dim of the generic embedding space")

"""training commmand epochs"""
parser.add_argument("--epochs",           default = 1000,      help = "number of epochs for the training")
parser.add_argument("--batch_size",       default = 10,        help = "the batch size")
parser.add_argument("--lr",               default = 1e-4,      help = "learning rate")
parser.add_argument("--ckpt_epochs",	  default = 10,       help = "for the epochs to save the checkpoints")
parser.add_argument("--ckpt_dir",		  default = "outputs/checkpoints")
parser.add_argument("--ckpt_name",		  default = "mifafa")

"""handle the textual token encoder"""
parser.add_argument("--corpus",           default = "data/corpus.txt")
parser.add_argument("--vocab_size",       default = 10000,     help = "number of vocabulary to hold in the text encoder")

"""handle the image mask object encoder"""
parser.add_argument("--num_channels",     default = 3,         help = "number of input channels of the image encoder")


parser.add_argument("--load_ckpt",  default = None,      help = "load the checkpoint")

config = parser.parse_args()
