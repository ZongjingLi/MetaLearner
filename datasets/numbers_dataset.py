# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 10:34:40
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 19:49:31
from helchriss.dsl.dsl_values import Value
from typing import List
from .base_dataset import SceneGroundingDataset

__all__ = ["get_dataset"]



# dataset of integer domain addition and max and min etc.
num_queries = ["two plus one", "two plus three", "one plus one", "red plus two",
               "three bigger than one",
               "three bigger than two",
               "one smaller than two",
               "one smaller than three",
               "one smaller than two plus three",
               ]
num_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0), Value("int", 3.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               ]
n_queries = len(num_queries)
num_groundings = [None] * n_queries

def get_dataset():return SceneGroundingDataset(num_queries, num_answers, groundings = num_groundings)
