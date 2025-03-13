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
from .metaphors.diagram_legacy import ConceptDiagram, MetaphorMorphism

"""structure for meta-learning of new domains"""
from .curriculum import MetaCurriculum
from .prompt.access_llm import run_gpt
from rinarak.domain import Domain
from rinarak.knowledge.executor import CentralExecutor

from rinarak.logger import get_logger, set_logger_output_file
from torch.utils.tensorboard import SummaryWriter
from domains.utils import load_domain_string, domain_parser

from itertools import tee

def is_subsequence(subseq, seq):
    """
    Check if subseq is a subsequence of seq
    """
    it = iter(seq)
    return all(any(x == y for y in it) for x in subseq)

def contains_subsequence(metaphor_path, metaphor_pattern):
    """
    Check if metaphor_path contains at least one subsequence in metaphor_pattern.
    
    :param metaphor_path: List of metaphor path elements
    :param metaphor_pattern: List of tuples, where each tuple represents a subsequence pattern
    :return: True if at least one pattern is a subsequence of metaphor_path, False otherwise
    """
    for pattern in metaphor_pattern:
        if is_subsequence(pattern, metaphor_path):
            return True
    return False

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
            "image" : ImageEncoder(generic_dim),
            "pointcloud" : PointCloudEncoder(generic_dim),
            "pointcloud_relation" : PointCloudRelationEncoder(generic_dim)
        })
        self.concept_diagram = ConceptDiagram()

        self.prompt = """core/prompt/metalearn_prompts.txt"""

        self.logger = get_logger("Citadel")
        set_logger_output_file("logs/citadel_logs.txt")
        self.device = "cuda:0" if torch.cuda.is_available() else "mps:0"

        self.concept_diagram.to(self.device)

    def to_dict(self):
        """Serialize the model architecture (excluding weights) for reconstruction."""
        return {
            "generic_dim": self.config.generic_dim,
            "vocab_size": self.config.vocab_size,
            "num_channels": self.config.num_channels,
            "encoder_types": list(self.encoders.keys()),  # Store encoder names
            "concept_diagram": self.concept_diagram.to_dict(),  # Store conscept diagram structure
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

    def train(self, ground_dataset):
        return self

    def evaluate(self, inputs, predicate, encoder_name = "PointcloudObjectEncoder", eval_mode = "literal"):
        outputs = {}
        source_state = self.encoders[encoder_name](inputs)

        result = self.concept_diagram.evaluate(source_state, predicate, None, eval_mode)
        return result

    def ground_batch(self, batch, metaphor_patterns = None,eval_mode="metaphor",  device = None):
        if not device : device = self.device
        root_name = self.concept_diagram.root_name # set the root of the evaluation path
        """only metaphor chain follows the given pattern will be optimized and the given pattern prob increase"""
        if not metaphor_patterns : metaphor_patterns = [(root_name,)]

        """initalize the batch loss and count the average metric over the whole batch"""
        batch_loss = 0.0
        batch_correct_count = 0
        input_states = batch["input"]
        batch_size = len(input_states)

        """iterate over all the input states over the batch """
        for i,scene in enumerate(input_states): # this is dump but some kind of batchwise operation
            
            scene = torch.stack(scene).to(device) # normally a nx... input scene
            scene_predicates = batch["predicate"]

            scene_count = 0
            scene_loss = 0.0
            
            for pred in scene_predicates: # iterate over all the predicates to evaluate 
                if pred == "end": break # #TODO: reserved words handle
                pred # the name fo the predicate
                gt = scene_predicates[pred][0][i].to(self.device) # tensor repr of the predicate

                result  = self.evaluate(scene, pred, encoder_name = "pointcloud", eval_mode = eval_mode)
                valid_path_count = 0 # the number of paths that matches the pattern
                avg_predicate_correct_vals = 0
                avg_predicate_loss = 0.0
                """
                each predicate evaluation have multiple measurement results and confidence.
                each predicate evaluation come with a path of evaluation (metaphor path)
                only evaluations with metaphor path that has certain pattern need to be optimized,
                confidence of measurements with those paths is also enhanced.
                """
                measure_results = result["results"]
                for i in range(len(measure_results)): 

                    #check if the metaphor path of the evalutation matches the pattern
                    metaphor_path = [root_name,]
                    for pat in result["metas_path"][i]:metaphor_path += [pat[1]]
                    path_control = contains_subsequence(metaphor_path, metaphor_patterns)
                   
                    # loss of grounding predicate
                    mask = ~torch.eye(result["results"][i].shape[-1], dtype=torch.bool, device=result["results"][i].device)  # Create a mask to ignore diagonal terms
                    avg_predicate_loss += torch.nn.BCEWithLogitsLoss()(result["results"][i][mask], gt[mask]) * path_control  # Apply mask

                    # loss of path locating path
                    avg_predicate_loss += -torch.log(result["probs"][i].reshape([])) * path_control # Path Strength Loss


                    gt_mat = (gt+0.5).int() # the ground truth boolean tensor
                    gt_prd = (result["results"][i].sigmoid() + 0.5).int() # predicted boolean tensor

                    #print(result["results"][i][mask].sigmoid())
                    #print(result["metas_path"]) #A list of metaphor paths
                    #print(result["symbol_path"]) #A list of symbol paths
                    #print(result["results"][i][mask].sigmoid())

                    valid_path_count += path_control # count the number of 
                    mask = ~torch.eye(gt_mat.shape[0], dtype=torch.bool, device=gt_mat.device)  # Create a mask to ignore diagonal terms
                    avg_predicate_correct_vals += path_control * (torch.sum((gt_mat[mask] == gt_prd[mask]))) / (gt_mat.shape[0] * (gt_mat.shape[1] - 1))
                    #print(path_control * (torch.sum((gt_mat == gt_prd))) / math.prod(list(gt_mat.shape)))

                avg_predicate_correct_vals /= valid_path_count
                avg_predicate_loss /= valid_path_count
         
                scene_count += avg_predicate_correct_vals
                scene_loss += avg_predicate_loss

            scene_count /=  (len(scene_predicates) - 1 )
            scene_loss /= (len(scene_predicates) - 1)
            #print(scene_count,  len(scene_predicates))

            batch_correct_count += scene_count
            batch_loss += scene_loss

        batch_loss /= batch_size
        batch_correct_count /= batch_size

        return batch_loss, batch_correct_count

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
        if curriculum.descriptive:
            pairings = curriculum.descriptive
        if meta: # if use meta learning approach, add some structure short cut using LLM
            questions = ["Explain to me what is Han-Banach theorem in metaphors."]
            pairings = run_gpt(questions, prompts)
        else:
            pass

        # add the target domain to the concept-diagram and add trivial connections
        root_name = self.concept_diagram.root_name
        if target_name not in self.concept_diagram.domains:
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

            if source!=root_name:
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
                    loss, count = self.ground_batch(batch, pairings)
                    train_count += count # normalize across the whole batch
                    train_loss += loss.detach()

                    optimizer.zero_grad()

                    #with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                """if the test loader is avaliable then do run evaluation on test set"""
                test_loss = 0.0
                test_count = 0
                if testloader: # just checking if the testing scenario actually exists
                    test_loss = 0.0
                    for batch in testloader:
                        batch_loss, count = self.ground_batch(batch, pairings)

                        batch_loss = batch_loss
                        test_count += count
                        test_loss += float(batch_loss)
                    writer.add_scalar("test_loss", test_loss / len(trainloader), epoch)
                    writer.add_scalar("test_percent", test_count /len(testloader), epoch)

                writer.add_scalar("train_loss", train_loss / len(trainloader), epoch)
                writer.add_scalar("train_percent", train_count /len(trainloader), epoch)
                if not(epoch % ckpt_itrs):save_ensemble_model(self, f"{config.ckpt_dir}/local_model.ckpt")

        """4) test the learning results on the given test data"""
        test_count = 0
        if testloader:
            for batch in tqdm(testloader):
                batch_loss, count = self.ground_batch(batch)
                batch_loss = batch_loss 
                test_loss += float(batch_loss)
                test_count += count 
        self.logger.info("Test Percent: {}".format(float(test_count) / len(testloader)))
        outputs = {}
        save_ensemble_model(self, f"{config.ckpt_dir}/{config.ckpt_name}.ckpt")
        return self


def curriculum_learning(config, model : EnsembleModel, meta_curriculums : List[MetaCurriculum]):
    model.logger.info("start the curriculum learning of the model")

    for curriculum in meta_curriculums:
        model.meta_learn_domain(config, curriculum)

def save_ensemble_model(model, path="checkpoints/ensemble_model.ckpt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save model structure
    model_config = model.to_dict()

    # Extract all domains from the concept diagram and save only domain strings
    domain_configs = {}
    for domain_name, domain_executor in model.concept_diagram.domains.items():
        domain_configs[domain_name] = domain_executor.domain.to_dict()  # Save domain string

    with open(path.replace(".ckpt", "_model.json"), "w") as f:
        json.dump(model_config, f)

    with open(path.replace(".ckpt", "_domains.json"), "w") as f:
        json.dump(domain_configs, f)  # Save all domain strings

    # Save model parameters & optimizer state
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    torch.save(checkpoint, path)


def load_ensemble_model(config, path="checkpoints/ensemble_model.ckpt"):
    # Load model structure
    with open(path.replace(".ckpt", "_model.json"), "r") as f:
        model_config = json.load(f)

    # Load all domain structures
    with open(path.replace(".ckpt", "_domains.json"), "r") as f:
        domain_configs = json.load(f)

    # Reconstruct all domains
    domains = {}
    for domain_name, domain_data in domain_configs.items():
        domains[domain_name] = CentralExecutor(load_domain_string(domain_data["domain_string"], domain_parser))

    # Reconstruct EnsembleModel
    model = EnsembleModel(config)

    # Restore encoders
    for encoder_type in model_config["encoder_types"]:
        if encoder_type not in model.encoders:
            raise ValueError(f"❌ Unknown encoder type: {encoder_type}")

    # Restore Concept Diagram and domains
    model.concept_diagram = ConceptDiagram()
    for domain_name, executor in domains.items():
        try:
            exec(f"from domains.{domain_name.lower()}.{domain_name.lower()}_domain import {executor.domain.domain_name}Domain")

            predef_domain = eval(f"{executor.domain.domain_name}Domain()")
            predef_domain.setup_predicates(executor)
            executor.visualize = predef_domain.visualize
        except:
            model.logger.warning(f"{domain_name}, not loaded")
        model.concept_diagram.add_domain(domain_name, executor)

    # Restore Morphisms
    for morphism_name, morphism_info in model_config["concept_diagram"]["morphisms"].items():
        source = morphism_info["source"]
        target = morphism_info["target"]
        morphism = MetaphorMorphism(
            model.concept_diagram.domains[source],
            model.concept_diagram.domains[target]
        )
        model.concept_diagram.add_morphism(source, target, morphism, name=morphism_name)

    # Load weights
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
