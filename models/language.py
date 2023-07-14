import torch
import torch.nn as nn
import torch.nn.functional as F

from .executor import *
from utils import *

class LanguageModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vectorize = VectorConstructor(config)
        self.vector_encoder = nn.GRU(config.word_dim,int(config.semantics_dim/2), batch_first =True, bidirectional=True) 
        

    def forward(self,x):
        repres = [self.vectorize.get_word_vectors(xs) for xs in x]
        seqes  = [self.vector_encoder(wvs)[0] for wvs in repres]

        return {"seq_features":seqes}

    def translate(self, inputs, lib, targets = None):
        """
        lib: [**{tokens,features,args}]
        """
        repres = [self.vectorize.get_word_vectors(xs) for xs in inputs]
        seqes_features = [self.vector_encoder(wvs)[0] for wvs in repres]

        if targets is not None:
            target_qs = [ConceptProgramExecutor.parse(target) for target in targets]
        else:target_qs = [None for _ in seqes_features]
        
        parse_results = []
        for i,feature in enumerate(seqes_features):
            self.decode(feature, lib, target_qs[i])
        outputs = {"loss":[0.0],"translate":["nil"]}
        return outputs

    def decode():
        return 0
class VectorConstructor(nn.Module):
    def __init__(self,config):
        super().__init__()
        # construct vector embeddings and other utils for given corpus
        corpus_path = config.corpus_path

        with open(corpus_path) as corpus_loaded:
            corpus = [t.strip() for t in corpus_loaded]
            self.corpus = corpus
            self.key_words = []
            self.token_to_id = build_vocab(corpus)
        self.id_to_token = reverse_diction(self.token_to_id)

        self.word_vectors = nn.Embedding(config.num_words,config.word_dim)

    def forward(self,sentence):
        return sentence
    
    def get_word_vectors(self,sentence):
        code = encode(tokenize(sentence),self.token_to_id)
        word_vectors = self.word_vectors(torch.tensor(code))
        return word_vectors

