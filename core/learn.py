import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from helchriss.logger import get_logger
from datasets.base_dataset import SceneGroundingDataset
from .model import MetaLearner
from helchriss.utils.data import ListDataset
from helchriss.knowledge.symbolic import Expression

from typing import Mapping, List, Tuple, Any
import itertools

def unbounded_cues(word : str) -> str : return "Any"

def optimal_schedule(dataset : SceneGroundingDataset, learned_vocab : List[str]):

    learned_vocab_set = set(learned_vocab)
    corpus = [entry["query"] for entry in dataset.data]
    
    all_words = set()
    for sentence in corpus:
        words = sentence.split()
        all_words.update(words)
    unlearned_words = all_words - learned_vocab_set
    """find the unlearned vocabulary (essentially newly learned words)"""

    # Map each dataset entry to the unlearned words it contains
    entry_to_unlearned = {}
    for i, entry in enumerate(dataset.data):
        sentence = entry["query"]
        words = set(sentence.split())
        unlearned_in_sentence = words - learned_vocab_set
        if unlearned_in_sentence:  # Only consider entries with unlearned words
            entry_to_unlearned[i] = unlearned_in_sentence
    
    # Keep track of optimal words to add
    best_words_to_add = set()
    max_new_learnable_entries = []

    
    # Start with k=1 (adding one word) and increase if needed
    k = 1
    improvement_found = False
    
    while not improvement_found and k <= len(unlearned_words):
        improvement_found = False
        
        # Try all combinations of k unlearned words
        for words_combo in itertools.combinations(unlearned_words, k):
            words_combo_set = set(words_combo)
            
            # Find which entries become learnable with these words
            newly_learnable_indices = []
            for entry_idx, unlearned_in_query in entry_to_unlearned.items():
                if unlearned_in_query.issubset(words_combo_set):
                    newly_learnable_indices.append(entry_idx)
            
            # If this is better than our current best, update
            if len(newly_learnable_indices) > len(max_new_learnable_entries):
                max_new_learnable_entries = [dataset.data[idx] for idx in newly_learnable_indices]
                best_words_to_add = words_combo_set
                improvement_found = True
        
        # If found an improvement at this k, no need to try larger k
        if improvement_found:
            break
        k += 1
    return list(best_words_to_add), max_new_learnable_entries

class AutoLearnSchedule:
    def __init__(self, dataset : SceneGroundingDataset, cues : Mapping[str, str] = None):
        self.dataset = dataset
        self.cues = unbounded_cues if not cues else cues
        self.base_vocab : List[str] = None
        self.logger = get_logger("AutoSchedule")

    def train(self, model : MetaLearner, epochs : int, lr : float = 2e-2):
        return model.train(self.dataset, epochs = epochs, lr = lr)
    
    def procedual_train(self, model : MetaLearner, eps = 0.001):
        base_vocab = []#model.learned_vocab
        base_data  = []
        base_dataset = ListDataset(base_data)
        step_epochs = 100

        new_words, _ = optimal_schedule(self.dataset, base_vocab)
        while new_words:
            new_words, slice_data = optimal_schedule(self.dataset, base_vocab)
            #for word in base_vocab:model.parser.purge_entry(word, 0.01, abs = 0)
            base_vocab.extend(new_words)
            base_data.extend(slice_data)
            self.logger.info(f"start to learn the words {new_words}, add corpus size {len(slice_data)}")
            [base_dataset.add(data) for data in slice_data]

            model, info = self.train_phase(model, base_dataset, epochs = step_epochs, eps = eps)
            avg_loss = info["loss"]
            
            self.logger.info(f"learned words : {new_words} avg_loss:{avg_loss}")
        for word in base_vocab:
            model.parser.purge_entry(word, 0.001, abs = 0)
            model.parser.display_word_entries(word)
        self.logger.info(f"complete the learning of words {base_vocab}")

    def train_phase(self, model : MetaLearner, slice_dataset, epochs : int = 100, eps : float = 0.01):
        done = False
        while not done:
            info = model.train(slice_dataset, epochs = epochs)
            avg_loss = info["loss"]
            if avg_loss < eps: done = True
        return model, {"loss" : avg_loss}
    
    #TODO: metaphorical expression infer

    def infer_metaphors(self, model : MetaLearner, slice_dataset, topK = 2):
        for data in slice_dataset:
            sentence = data["query"]
            maximal_parse = model.maximal_parse(sentence)
            for parse in maximal_parse[:topK]:
                expr = Expression.parse(str(parse[0].sem_program))
                model.infer_metaphor_expressions(expr)
        return model