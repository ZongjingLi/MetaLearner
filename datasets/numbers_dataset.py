from helchriss.dsl.dsl_values import Value
from typing import List
from .base_dataset import SceneGroundingDataset



# dataset of integer domain addition and max and min etc.
num_queries = ["two plus one", "two plus three", "one plus one", "red plus two"]
num_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0), Value("int", 3.0)]
n_queries = len(num_queries)
num_groundings = [None] * n_queries

def get_dataset():return SceneGroundingDataset(num_queries, num_answers, groundings = num_groundings)