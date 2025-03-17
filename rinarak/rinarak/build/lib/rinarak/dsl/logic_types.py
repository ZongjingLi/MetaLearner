'''
 # @ Author: Zongjing Li
 # @ Create Time: 2023-11-28 04:10:30
 # @ Modified by: Zongjing Li
 # @ Modified time: 2023-12-26 23:14:21
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
from rinarak.types import *
from rinarak.program import *

# [Type Specification] of ObjectSet, Attribute, Boolean and other apsects
Stream = baseType("Stream")
fuzzy_set = baseType("ObjectSet")
PrimitiveSet = baseType("PrimitiveSet")
attribute = baseType("Attribute")
boolean = baseType("Boolean")
Concept = baseType("Concept")
integer = baseType("Integer")

from rinarak.utils.tensor import logit, expat
