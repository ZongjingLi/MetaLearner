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

import numpy as np
import os

from .encoders.image_encoder import ImageEncoder
from .encoders.text_encoder  import TextEncoder

from .metaphors.diagram import ConceptDiagram

class EnsembleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        generic_dim = int(config.generic_dim)
        sequences = []
        with open(config.corpus) as corpus:
            for line in corpus:
                line = line.strip()
                if line:
                    line = line.lower()
                    line = ' '.join(line.split())
                    sequences.append(line)
        

        """general domain encoder (image/text)"""
        self.encoders = {
            "text": TextEncoder(
                generic_dim, vocab_size = int(config.vocab_size),
                sequences = sequences, punct_to_remove=['.', '!', ',']),
            "image" : ImageEncoder(generic_dim, config.num_channels),
        }
        self.concept_diagram = ConceptDiagram()
    
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

    def train(self, ground_dataset):
        return self