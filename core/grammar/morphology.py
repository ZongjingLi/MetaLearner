import torch
import torch.nn as nn

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet

# Download required data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Stemming
stemmer = PorterStemmer()
print(stemmer.stem("running"))  # "run"

# Lemmatization
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("children", pos='n'))  # "child"
print(lemmatizer.lemmatize("running", pos='v'))   # "run"