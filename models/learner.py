import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn import *
from .executor  import *
from .language  import *
from .reasoning import *
import networkx as nx

class VanillaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, quries):
        return quries

class MetaLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        # [Concept Structure Embedding]
        self.box_registry = build_box_registry(config)
        self.entailment = build_entailment(config)

        # [Language Semantics Encoder]
        self.language_encoder = LanguageModel(config)

        # [Regular Program Searcher]
        self.token_embeddings = torch.nn.Embedding(config.num_tokens, config.token_dim)
        self.tokens = [config.translator[clsx].__name__ for clsx in config.translator]
        self.q_args = [config.args_num[clsx] for clsx in config.translator]
        self.concept2token = nn.Linear(config.concept_dim * 2,config.token_dim)

        # [Neuro Symbolic Executor]
        self.executor = ConceptProgramExecutor(config)
        self.rep = config.concept_type
        self.applier = None
        self.p = .97

        # [Neuro Predicate Plan Search]
        self.neuro_reasoner = NeuroReasoner()
    
    def get_token_embeddings(self):return self.token_embeddings(torch.tensor(list(range(len(self.tokens)))))

    def get_concept_embeddings(self):return 0

    def forward(self, quries):
        return quries
    
    def translate(self, statements, programs = None):
        
        libs = list()
        for b in range(len(statements)):
            # [Make Program Token Embeddings]
            q_tokens = self.tokens
            q_features = self.get_token_embeddings()
            q_args = self.q_args

            # [Make Concept Token Embeddings]
            c_tokens = self.executor.concept_vocab
            c_features = self.concept2token(torch.cat([self.executor.get_concept_embedding(c) for c in c_tokens],dim =0))
            c_args = [0 for _ in range(len(c_tokens))]

            # [Merge Features from different domains]
            all_tokens = list();all_tokens.extend(q_tokens);all_tokens.extend(c_tokens)
            all_args = list();all_args.extend(q_args);all_args.extend(c_args)
            all_features = torch.cat([q_features, c_features], dim = 0)

            libs.append({"tokens":all_tokens,
                        "features":all_features,
                        "args_num":all_args})

        outputs = self.language_encoder.translate(statements,libs,self.executor,programs)
        return outputs
    
    def search(self, init_state, init_latent, goal):
        """
        args:
            init_state: 
            init_latent:
            goal: the goal statement of search. The search ends when the goal statement is evaluated as true.
        outputs:
            the search path and the corresponding confidence
        """
        open_states = PriorityQueue()
        close_states = set()

        for possible_state in self.applier.expand(init_state):
            if self.applier.evaluate(possible_state, goal) >= self.p:
                print("solved")
            pass

        return 0