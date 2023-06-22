import argparse

parser = argparse.ArgumentParser()

# language model configuration
parser.add_argument("--corpus_path",            default = "assets/corpus.txt")

# reasoning model configuration
parser.add_argument("--box_dimension")

config = parser.parse_args(args = [])