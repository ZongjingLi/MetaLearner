# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:43:08
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-25 02:20:09

import torch
import torch.nn as nn

import numpy as np

from rinarak.dklearn.nn import FCBlock, ConvolutionUnits

class BackwardSearchPlanner:
	def __init__(self, config):
		self.config = config

class GoalRegressionPlanner:
	def __init__(self, config):
		self.config = config