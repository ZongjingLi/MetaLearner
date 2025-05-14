# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-05-03 18:47:15
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-05-03 18:51:29
from helchriss.dsl.dsl_values import Value
from typing import List, Mapping, Dict
from datasets.base_dataset import SceneGroundingDataset


# dataset of integer domain addition and max and min etc.
group = [
    "A group is a set with operation and identity",
    "three is greater than one",
    "three is greater than one",
    "any group with element e1 and e2"]
group_answers = [
    Value("boolean",0.0),
    Value("boolean",1.0),
    Value("boolean", 1.0)]
n_queries = len(group)
num_groundings = [None] * n_queries

def get_dataset():return SceneGroundingDataset(group, group_answers, groundings = num_groundings)