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
from rinarak.knowledge.executor import CentralExecutor

from rinarak.logger import get_logger, set_logger_output_file
from torch.utils.tensorboard import SummaryWriter




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
        self.encoders = nn.ModuleDict({
            "text": TextEncoder(
                generic_dim, vocab_size = int(config.vocab_size),
                sequences = sequences, punct_to_remove=['.', '!', ',']),
            "image" : ImageEncoder(generic_dim, config.num_channels),
            "pointcloud" : PointCloudEncoder(generic_dim),
            "pointcloud_relation" : PointCloudRelationEncoder(generic_dim)
        })
        self.concept_diagram = ConceptDiagram()

        self.prompt = """core/prompt/metalearn_prompts.txt"""

        self.logger = get_logger("Citadel")
        set_logger_output_file("logs/citadel_logs.txt")
        self.device = "cuda:0" if torch.cuda.is_available() else "mps:0"

        self.concept_diagram.to(self.device)
    
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

        result = self.concept_diagram.evaluate(source_state, predicate, None, eval_mode)

        return result

    def ground_batch(self, batch, device = None):
        if not device : device = self.device
        batch_loss = 0.0
        correct_count = 0
        batch_size = len(batch["input"])
        for i,scene in enumerate(batch["input"]): # this is dump but some kind of batchwise operation
            scene = torch.stack(scene).to(device) # normally a nx... input scene
            for pred in batch["predicate"]:
                if pred == "end": break
                pred # the name fo the predicate
                gt = batch["predicate"][pred][0][i].to(self.device) # tensor repr of the predicate

                result  = self.evaluate(scene, pred, encoder_name = "pointcloud")
                for i in range(len(result["results"])):
                    batch_loss += torch.nn.MSELoss()(gt, result["results"][i])

                    gt_mat = (gt+0.5).int()
                    gt_prd = (result["results"][i] + 0.5).int()
       
                    correct_count += (torch.sum((gt_mat == gt_prd))) / math.prod(list(gt_mat.shape))
        batch_loss /= len(result["results"])
        return batch_loss, correct_count

    def meta_learn_domain(self, config, curriculum : MetaCurriculum, epochs = 1000, lr = 2e-4, meta = False):
        """1) create the template domain executor or use a custom domain executor"""
        target_domain = curriculum.concept_domain # get the concept domain from the curriculum
        self.to(self.device)

        assert isinstance(target_domain, Domain), "the dmomain file fo the executor is not actual Domain"
        target_name = target_domain.domain_name
        target_executor =  CentralExecutor(target_domain, "cone", 256)


        """2) extract the metaphorical enailment relations"""

        with open(self.prompt) as f:
            prompts_str = f.read()
            system_prmopt, user_prompt = prompts_str.split('----')
            prompts = {
                'system': system_prmopt.strip(),
                'user': user_prompt.strip()
            }
        if meta: # if use meta learning approach, add some structure short cut using LLM
            questions = ["Explain to me what is Han-Banach theorem in metaphors."]
            pairings = run_gpt(questions, prompts)
        else:
            pairings = []

        # add the target domain to the concept-diagram and add trivial connections
        root_name = self.concept_diagram.root_name
        self.concept_diagram.add_domain(target_name, target_executor)
        self.concept_diagram.add_morphism(root_name, target_name, MetaphorMorphism(
            self.concept_diagram.domains[root_name], self.concept_diagram.domains[target_name]
            ))
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
                assert False, "Not implemented the predicate source target pairing."

        """3) train the concept-diagram and the encoding """
        from datasets.scene_dataset import scene_collate
        from tqdm import tqdm
        epochs = int(config.epochs)
        batch_size = int(config.batch_size)
        ckpt_epochs = int(config.ckpt_epochs)
        lr = float(config.lr)
        writer = SummaryWriter("logs")

        optimizer = torch.optim.Adam(self.parameters(), lr = lr)
        
        trainloader = DataLoader(curriculum.train_data, batch_size, collate_fn = scene_collate, shuffle = True)
        if curriculum.test_data:
            testloader = DataLoader(curriculum.test_data, batch_size, collate_fn = scene_collate, shuffle = True)

        if trainloader:
            ckpt_itrs = config.ckpt_epochs
            for epoch in tqdm(range(epochs)):
                train_loss = 0.0
                train_count = 0
                for batch in trainloader:
                    loss, count = self.ground_batch(batch)
                    train_count += count / batch_size
                    loss = loss / batch_size # normalize across the whole batch
                    train_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                """if the test loader is avaliable then do run evaluation on test set"""
                test_loss = 0.0
                test_count = 0
                if testloader: # just checking if the testing scenario actually exists
                    test_loss = 0.0
                    test_count = 0
                    for batch in testloader:
                        batch_loss, count = self.ground_batch(batch)
                        batch_loss = batch_loss / batch_size
                        test_loss += float(batch_loss)
                        test_count += count / batch_size
                    writer.add_scalar("test_loss", test_loss / len(trainloader), epoch)
                    writer.add_scalar("test_percent", test_count /len(trainloader), epoch)

                writer.add_scalar("train_loss", train_loss / len(trainloader), epoch)
                writer.add_scalar("train_percent", train_count /len(trainloader), epoch)
                if not(epoch % ckpt_itrs):torch.save(self.state_dict(),f"{config.ckpt_dir}/local_model.ckpt")

        """4) test the learning results on the given test data"""
        
        if testloader:
            for batch in tqdm(testloader):
                batch_loss, count = self.ground_batch(batch)
                batch_loss = batch_loss / batch_size
                test_loss += float(batch_loss)
                test_count += count / batch_size
        print("Test:",test_count / len(testloader))
        outputs = {}
        return self


def curriculum_learning(config, model : EnsembleModel, meta_curriculums : List[MetaCurriculum]):
    model.logger.info("start the curriculum learning of the model")

    for curriculum in meta_curriculums:
        model.meta_learn_domain(config, curriculum)