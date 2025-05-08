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
size_queries = ["two plus one", "two plus three", "one plus one", "red plus two",
               "three bigger one",
               "three smaller one",
               "three bigger two",
               "one smaller two",
               "one smaller three",
               "one bigger three",
               "one bigger two",
               "three plus two",
               "three plus one",
               "three plus three",
               ]
size_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0), Value("int", 3.0),
               Value("boolean", 1.0),
               Value("boolean", 0.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               Value("boolean", 1.0),
               Value("boolean", 0.0),
               Value("boolean", 0.0),
               Value("int", 5.0),
               Value("int", 4.0), 
               Value("int", 6.0), 
               ]
n_queries = len(size_queries)
num_groundings = [None] * n_queries

def get_dataset():return SceneGroundingDataset(size_queries, size_answers, groundings = num_groundings)
