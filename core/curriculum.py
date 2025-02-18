# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 05:45:31
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-08 19:54:48
import torch
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from rinarak.domain import load_domain_string
from domains.utils import domain_parser
from rinarak.knowledge.executor import CentralExecutor
from typing import List, Optional
import re
import ast

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
            self.concept_domain = load_domain_string(concept_domain, domain_parser)
        
        """2) create the dataset for the domain to learn. Can considered as pure grounding dataset or learnd by other methods"""
        self.train_data = train_data if isinstance(train_data, Dataset) else None
        assert self.train_data is not None, "Train data must be a valid Dataset instance"


        """3) some desciptive sentences that gives the enailment relation between the source domain predicates and target domain"""
        self.descriptive = descriptive  # List of descriptive sentences

        #assert isinstance(self.descriptive, list), "Descriptive must be a list of sentences"

        """4) similar to previous step load the test cases for compositional learning"""
        self.test_data = test_data if isinstance(test_data, Dataset) else None

    def evaluate(self, model: nn.Module):
        if not self.test_loader:
            print("No test data provided.")
            return

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, labels = batch
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")


def _parse_curriculum(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    curricula = re.split(r'<Curriculum>', content)[1:]
    parsed_curricula = []
    
    for curriculum in curricula:
        sections = re.split(r'<([^>]+)>', curriculum)[1:]
        parsed_data = {}
        for i in range(0, len(sections), 2):
            section_name = sections[i].strip()
            section_content = sections[i+1].strip()
            
            if section_name in ['TrainData', 'TestData']:
                parsed_data[section_name] = section_content.split('\n')
            else:
                parsed_data[section_name] = section_content
        
        parsed_curricula.append(parsed_data)
    
    return parsed_curricula

def load_curriculum(file_path):
    parsed_curricula = _parse_curriculum(file_path)
    outputs = []
    for i in range(len(parsed_curricula)):
        concept_domain = parsed_curricula[i]["ConceptDomain"] # Domain File

        train_data_commands = parsed_curricula[i]["TrainData"] # Load Train Data
        for line in train_data_commands[:-1]:exec(line)
        train_dataset = eval(train_data_commands[-1])

        test_data_commands = parsed_curricula[i]["TestData"]
        for line in test_data_commands[:-1]:exec(line)
        test_dataset = eval(test_data_commands[-1]) # Load Test Data
        
        descriptive = parsed_curricula[i]["Metaphor"] # Metaphors
        descriptive = ast.literal_eval( re.sub(r'(\w+)', r'"\1"', descriptive) )

        outputs.append(MetaCurriculum(concept_domain, train_dataset, descriptive, test_dataset))
    return outputs

