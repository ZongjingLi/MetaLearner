# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-19 04:42:09
import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

from rinarak.dklearn.nn import FCBlock
from rinarak.utils.tensor import logit

class State:
    def __init__(self, data):
        """ construct a symbolic or hybrid state
        Args:
            data: a diction that maps the data of each state to the actual value
        """
        self.data = data
    
    def get(self, predicate_name): return self.data[predicate_name]

class Precondition:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self, bool_expression) -> bool:
        return 0
    
    def __str__(self): return str(self.bool_expression)

class Effect:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self) -> bool:
        return 0
    
    def __str__(self):return str(self.bool_expression)
    
    def split_effect(self):
        """split the effect into two parts. effect+, effect-
        Returns:
            effect+: the added predicates to the known state
            effect-: the removed predicates to the known state
        """
        return 0

class Action:
    def __init__(self, action_name, parameters, precondition, effect):
        """ construct an symboic action with preconditinos and effects
        Args:
            action_name: the name of the action
            precondition: a boolean expression that is callable for an input predicate state
            effect: a set of assignment expressions to known predicates
        """
        super().__init__()
        self.action_name = action_name
        self.parameters = parameters
        if isinstance(precondition, Precondition):
            self.precondition = precondition
        else: self.precondition = Precondition(precondition)

        if not isinstance(effect, Effect):
            self.effect = Effect(effect)
        else: self.effect = effect
    
    def apply(self, state):
        if self.precondition(state):
            return

class Expression(nn.Module):
    def __init__(self, expression_nested):
        super().__init__()
        self.expression_nested = expression_nested
    
    def evaluate(self, inputs, executor):
        return inputs

class PredicateFilter(nn.Module):
    def __init__(self, concept, arity = 1):
        super().__init__()
        self.concept = concept
    
    def __str__(self): return self.concept

    def __repr__(self): return self.__str__()
    
    def forward(self, x):
        executor = x["executor"]
        concept = self.concept
        features = x["features"]

        filter_logits = torch.zeros([1])
        parent_type = executor.get_type(concept)
        for candidate in executor.type_constraints[parent_type]:
            filter_logits += executor.entailment(features,
            executor.get_concept_embedding(candidate)).sigmoid()

        div = executor.entailment(features,
                executor.get_concept_embedding(concept)).sigmoid()
        filter_logits = logit(div / filter_logits)
        return{"end":filter_logits, "executor":x["executor"]}