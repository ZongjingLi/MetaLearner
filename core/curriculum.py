import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rinarak.domain import load_domain_string
from rinarak.knowledge.executor import CentralExecutor
from typing import List, Optional

class MetaCurriculum:
    """
    A meta learning cuuriculm is described by the following tuple (c, Xc, Dc, Tc)
        c : is the new concept domain to learn. 
        Xc : input cases paired with ground truth experiments.
        Dc : descriptive sentences that connects the source domain with the target domain
        Tc : Test cases for the new concepts, possibly ood
    """
    def __init__(self, concept_domain, train_data, descriptive, test_data = None):
        super().__init__()
        assert isinstance(concept_domain, CentralExecutor) or isinstance(concept_domain, str),\
              "input concept domain must be an already defined executor or a pddl domain string"
        
        """1) set the 'c' executor as the new domain to learn"""
        if isinstance(concept_domain, CentralExecutor):
            self.concept_domain = concept_domain
        if isinstance(concept_domain, str):
            self.concept_domain = load_domain_string(concept_domain)
        
        """2) create the dataset for the domain to learn. Can considered as pure grounding dataset or learnd by other methods"""

        """3) some desciptive sentences that gives the enailment relation between the source domain predicates and target domain"""

        """4) similar to previous step load the test cases for compositional learning"""
