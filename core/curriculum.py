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
        self.train_data = train_data if isinstance(train_data, Dataset) else None
        assert self.train_data is not None, "Train data must be a valid Dataset instance"
        self.train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)

        """3) some desciptive sentences that gives the enailment relation between the source domain predicates and target domain"""
        self.descriptive = descriptive  # List of descriptive sentences
        assert isinstance(self.descriptive, list), "Descriptive must be a list of sentences"

        """4) similar to previous step load the test cases for compositional learning"""
        self.test_data = test_data if isinstance(test_data, Dataset) else None
        self.test_loader = DataLoader(self.test_data, batch_size=32, shuffle=False) if self.test_data else None
    
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

def load_curriculum_string(string : str) -> MetaCurriculum:
    import xml.etree.ElementTree as ET
    root = ET.fromstring(string)

    concept_domain = root.find('ConceptDomain').text.strip()
    train_data_code = root.find('TrainData').text.strip()
    test_data_code = root.find('TestData').text.strip()
    descriptive = [(m.text.strip()) for m in root.findall('Metaphor')]
    
    exec(train_data_code)
    exec(test_data_code)

    return MetaCurriculum(concept_domain, TrainDataset(), descriptive, TestDataset())