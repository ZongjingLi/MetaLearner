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
from typing import List, Optional

"""a set of encoder that encode different modalities to the generic domain"""
from .encoders.image_encoder import ImageEncoder
from .encoders.text_encoder  import TextEncoder
from .encoders.pointcloud_encoder import PointCloudEncoder, PointCloudRelationEncoder

"""the backend neuro-symbolic concept learner for execution of predicates and actions"""
from .metaphors.diagram import ConceptDiagram, MetaphorMorphism

"""structure for meta-learning of new domains"""
from .curriculum import MetaCurriculum
from .prompt.access_llm import run_gpt
from rinarak.domain import Domain

from rinarak.logger import get_logger, set_logger_output_file


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
            "pointcloud" : PointCloudEncoder(generic_dim),
            "pointcloud_relation" : PointCloudRelationEncoder(generic_dim)
        }
        self.concept_diagram = ConceptDiagram()

        self.prompt = """core/prompt/metalearn_prompts.txt"""

        self.logger = get_logger("Citadel")
        set_logger_output_file("logs/citadel_logs.txt")
    
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

    def evaluate(self, inputs, predicate, encoder_name = "PointcloudObjectEncoder", eval_mode = "literal"):
        outputs = {}
        source_state = self.encoders[encoder_name](inputs)

        result = self.concept_diagram.evaluate(source_state, predicate, "GenericDomain", eval_mode)

        return outputs

    def batch_evaluate(self, inputs, predicate):
        return 
    
    def meta_learn_domain(self, curriculum : MetaCurriculum, epochs = 1000, lr = 2e-4):
        """1) create the template domain executor or use a custom domain executor"""
        target_executor = curriculum.concept_domain # get the concept domain from the curriculum
        assert isinstance(target_executor.domain, Domain), "the dmomain file fo the executor is not actual Domain"
        target_name = target_executor.domain.domain_name

        """2) extract the metaphorical enailment relations"""
        with open(self.prompt) as f:
            prompts_str = f.read()
            system_prmopt, user_prompt = prompts_str.split('----')
            prompts = {
                'system': system_prmopt.strip(),
                'user': user_prompt.strip()
            }
        questions = ["Explain to me what is Han-Banach theorem in metaphors."]
        pairings = run_gpt(questions, prompts)

        # add the target domain to the concept-diagram and add trivial connections
        root_name = self.concept_diagram.root_name
        self.concept_diagram.add_domain(target_name, target_executor)
        self.concept_diagram.add_morphism(root_name, target_name)
        # add other 'short-cuts' that is not from the generic domain to thte target-domain
        for metaphor_pair in pairings:
            source, target = metaphor_pair[0], metaphor_pair[1]
            #TODO: Check for is there any valid morphsims , if not exists, create one, else use the known metaphor
            ref_morphism = None if None else \
                MetaphorMorphism(
                    self.concept_diagram.domains[source],
                    self.concept_diagram.domains[target])
            self.concept_diagram.add_morphism(source, target, ref_morphism)
            if len(metaphor_pair) == 4:
                pass

        """3) train the concept-diagram and the encoding """
        trainloader = None
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        for epoch in range(epochs):
            for sample in trainloader:
                self.concept_diagram.evaluate()

                loss = 0.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        """4) test the learning results on the given test data"""
        testloader = None
        for sample in testloader:
            pass

        outputs = {}
        return 


def curriculum_learning(model : EnsembleModel, meta_curriculums : List[MetaCurriculum]):
    model.logger.info("start the curriculum learning of the model")
    for curriculum in meta_curriculums:
        model.meta_learn_domain(curriculum)