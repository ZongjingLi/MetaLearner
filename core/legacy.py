'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-19 10:04:08
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-19 10:05:02
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import math
import json
from typing import List, Optional

"""a set of encoder that encode different modalities to the generic domain"""
from .encoders.image_encoder import ImageEncoder
from .encoders.text_encoder  import TextEncoder
from .encoders.pointcloud_encoder import PointCloudEncoder, PointCloudRelationEncoder

"""the backend neuro-symbolic concept learner for execution of predicates and actions"""
from .metaphors.diagram_executor import MetaphorExecutor

"""the CCG based grammar learner"""

"""structure for meta-learning of new domains"""
from .curriculum import MetaCurriculum
from .prompt.access_llm import run_gpt
from helchriss.logger import set_logger_output_file, get_logger
from helchriss.utils import load_corpus

class EnsembleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "mps:0"
        self.config = config
        concept_dim = 128
        generic_dim = int(config.generic_dim) # the generic embedding space of all symbolic concepts
        
        sequences = load_corpus(config.corpus)

        """general domain encoder (image/text)"""
        self.encoders = nn.ModuleDict({
            "text": TextEncoder(
                generic_dim, vocab_size = int(config.vocab_size),
                sequences = sequences, punct_to_remove=['.', '!', ',']),
            "image" : ImageEncoder(generic_dim),
            "pointcloud" : PointCloudEncoder(generic_dim),
            "pointcloud_relation" : PointCloudRelationEncoder(generic_dim)
        })
        self.executor = MetaphorExecutor([],concept_dim = concept_dim)



        self.logger = get_logger("Citadel")
        set_logger_output_file("logs/citadel_logs.txt")
        



    def to_dict(self):
        """Serialize the model architecture (excluding weights) for reconstruction."""
        return {
            "generic_dim": self.config.generic_dim,
            "vocab_size": self.config.vocab_size,
            "num_channels": self.config.num_channels,
            "encoder_types": list(self.encoders.keys()),  # Store encoder names
            "concept_diagram": self.executor.to_dict(),  # Store conscept diagram structure
        }    

    def forward(self, inputs):
        return 
    
    def encode_image_scene(self, image, masks):
        """
        Encode image with multiple object masks.
        
        Args:
            image: Image tensor of shape (B, C, H, W)
            masks: List of mask tensors, each of shape (B, 1, H, W)
            
        Returns:
            Tensor of shape (B, num_masks, generic_dim) containing embeddings
            for each masked region
        """
        batch_size = image.shape[0]
        embeddings = []
        
        # Process each mask separately
        for mask in masks:
            # Get embedding for current mask
            embedding = self.encoders['image'](image, mask)
            # Project to shared space
            embedding = self.image_projection(embedding)
            # Normalize embedding
            embedding = self.layer_norm(embedding)
            embeddings.append(embedding)
        
        # Stack all embeddings (B, num_masks, D)
        scene_embedding = torch.stack(embeddings, dim=1)
        return scene_embedding

    def encode_text(self, text):
        """
        Encode text input.
        
        Args:
            text: String or list of strings to encode
            
        Returns:
            Tensor of shape (B, generic_dim) containing text embeddings
        """
        embeddings = self.encoders['text'].encode_text(text)
        return embeddings
