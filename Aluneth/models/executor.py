import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models.nn import build_entailment, build_box_registry
from Karanir.utils import *

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

class SceneGraphRepresentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.effective_level = 1
        self.max_level = 4

    @property
    def top_objects(self):
        return 0

class ConceptProgramExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.entailment = build_entailment(config)
        self.concept_registry = build_box_registry(config)

        # [Word Vocab]
        concept_vocab = []
        with open(config.root+"/knowledge/{}_concept_vocab.txt".format(config.domain)) as vocab:
            for concept_name in vocab:concept_vocab.append(concept_name.strip())

        self.concept_vocab = concept_vocab
        
        # args during the execution
        self.kwargs = None 

        # Hierarchy Representation
        self.hierarchy = 0

        self.translator = config.translator
    def add_concept_embedding(self, concept_name):
        self.concept_vocab.append(concept_name)

    def get_concept_embedding(self,concept):
        try:
            concept_index = self.concept_vocab.index(concept)
            idx = torch.tensor(concept_index).unsqueeze(0).to(self.config.device)
            return self.concept_registry(idx)
        except:
            print(concept_index)
            raise UnknownConceptError

    def forward(self, q, **kwargs):
        self.kwargs = kwargs

        return q(self)

    def parse(self,string, translator = None):
        string = string.replace(" ","")
        if translator == None: translator = self.translator
        def chain(p):
            head, paras = head_and_paras(p)
            if paras == None:
                q = head
            elif '' in paras:
                q = translator[head]()
            else:
                args = [chain(o) for o in paras]
                if head in translator: q = translator[head](*args)
                else: raise UnknownArgument
            return q
        program = chain(string)
        return program
