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
    corpus = [entry[1]["query"] for entry in dataset]
    
    all_words = set()
    for sentence in corpus:
        words = sentence.split()
        all_words.update(words)
    unlearned_words = all_words - learned_vocab_set
    """find the unlearned vocabulary (essentially newly learned words)"""

    # Map each dataset entry to the unlearned words it contains
    entry_to_unlearned = {}
    for i, entry in dataset:
        sentence = entry["query"]
        words = set(sentence.split())
        unlearned_in_sentence = words - learned_vocab_set
        if unlearned_in_sentence:  # Only consider entries with unlearned words
            entry_to_unlearned[i] = unlearned_in_sentence
    
    # Keep track of optimal words to add
    best_words_to_add = set()
    max_new_learnable_entries = []

    
    # Start with k=1 (adding one word) and increase if needed
    k = len(all_words)
    improvement_found = False
    
    while not improvement_found and k <= len(unlearned_words):
        improvement_found = False
        
        # try all combinations of k unlearned words
        for words_combo in itertools.combinations(unlearned_words, k):
            words_combo_set = set(words_combo)
            
            # Find which entries become learnable with these words
            newly_learnable_indices = []
            for entry_idx, unlearned_in_query in entry_to_unlearned.items():
                if unlearned_in_query.issubset(words_combo_set):
                    newly_learnable_indices.append(entry_idx)
            
            # if this is better than our current best, update
            if len(newly_learnable_indices) > len(max_new_learnable_entries):
                max_new_learnable_entries = [dataset[idx] for idx in newly_learnable_indices]
                best_words_to_add = words_combo_set
                improvement_found = True
        
        # if found an improvement at this k, no need to try larger k
        if improvement_found:
            break
        k += 1
    return list(best_words_to_add), max_new_learnable_entries

class AutoLearnSchedule:
    def __init__(self, dataset : SceneGroundingDataset, testset:SceneGroundingDataset, cues : Mapping[str, str] = None):
        self.dataset = dataset
        self.testset = testset
        self.cues = unbounded_cues if not cues else cues
        self.base_vocab : List[str] = None
        self.logger = get_logger("AutoSchedule")
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, model : MetaLearner, epochs : int, lr : float = 2e-2):
        return model.train(self.dataset, epochs = epochs, lr = lr)
    
    def procedual_train(self, model : MetaLearner, lr = 2e-4, eps = 0.005, verbose = False):
        device = self.device
        base_vocab = []#model.learned_vocab
        base_data  = []
        base_dataset = ListDataset(base_data)
        step_epochs = 5
        #model = model.to(device)

        new_words, _ = optimal_schedule(self.dataset, base_vocab)
        while new_words:
            new_words, slice_data = optimal_schedule(self.dataset, base_vocab)
            #for word in base_vocab:model.parser.purge_entry(word, 0.01, abs = 0)
            base_vocab.extend(new_words)
            base_data.extend(slice_data)
            self.logger.info(f"start to learn the words {new_words}, add corpus size {len(slice_data)}")
            [base_dataset.add(data) for data in slice_data]

            model, info = self.train_phase(model, base_dataset, epochs = step_epochs, eps = eps, lr = lr)
            avg_loss = info["loss"]
            avg_acc = info["acc"]
            
            self.logger.info(f"learned words : {new_words} avg_acc:{avg_acc} avg_loss:{avg_loss}")
        
        if verbose:
            for word in base_vocab:
                model.parser.purge_entry(word, 0.0001, abs = 0)
                model.parser.display_word_entries(word)
        self.logger.info(f"complete the learning of words {base_vocab}")

    def train_phase(self, model : MetaLearner, slice_dataset, epochs : int = 10, lr : int = 5e-4, eps : float = 0.005):
        done = False
        # one epoch with suprression just to add the connections
        self.logger.critical("unification structure check, eval run")
        #with torch.no_grad():
        info = model.train(slice_dataset, epochs = 1, lr = lr, unify = True)

        self.logger.critical("training until converges")
        while not done:
            info = model.train(slice_dataset, epochs = epochs, lr = lr, unify = False, test_set = self.testset)
            avg_loss = info["loss"]
            avg_acc = info["acc"]
            if 1 - avg_acc < eps: done = True
        return model, {"loss" : avg_loss, "acc": avg_acc}
    
    #TODO: metaphorical expression infer

    def infer_metaphors(self, model : MetaLearner, slice_dataset, topK = 2):
        ### infer cmt expressions from the given data set by checking type mismatch.
        for data in slice_dataset:
            sentence = data["query"]
            exprs = model.maximal_parse(sentence, forced = 1)[:topK]
            metas = model.infer_metaphor_expressions([expr[0] for expr in exprs])

        return model
    
