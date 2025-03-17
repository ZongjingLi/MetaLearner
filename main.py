'''
 # @ Author: Zongjing Li
 # @ Create Time: 2025-01-17 09:43:38
 # @ Modified by: Zongjing Li
 # @ Modified time: 2025-01-17 09:47:01
 # @ Description: This file is distributed under the MIT license.
'''
import os
import sys
import regex
import torch
import torch.nn as nn
import numpy as np
from rinarak.logger import get_logger, set_output_file
#from datasets.scene_dataset import SceneDataset
from datasets.scene_dataset import SceneDataset, DataLoader, scene_collate
from tqdm import tqdm
from core.model import save_ensemble_model, load_ensemble_model

main_logger = get_logger("Main")
set_output_file("logs/main_logs.txt")

domain_str = """
(domain Contact)
(:type
    state - vector[float, 256]        ;; [x, y] coordinates
)
(:predicate
    ;; Basic position predicate
    ref ?x-state -> boolean
    get_position ?x-state -> vector[float, 2]
    
    ;; Qualitative distance predicates
    contact ?x-state ?y-state -> boolean
)
"""
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
from domains.utils import domain_parser
import math


def train_grounding(config, model, train_scene_dataset : 'SceneDataset', test_scene_dataset = None):
    epochs = int(config.epochs)
    batch_size = int(config.batch_size)
    ckpt_epochs = int(config.ckpt_epochs)
    writer = SummaryWriter("logs")
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    trainloader  = DataLoader(train_scene_dataset, batch_size, collate_fn = scene_collate,shuffle = True)
    if test_scene_dataset is not None:
        testloader  = DataLoader(test_scene_dataset, batch_size, collate_fn = scene_collate,shuffle = True)
    else: testloader = None
    for epoch in tqdm(range(epochs)):
        train_loss = 0.0
        train_count = 0
        for batch in trainloader:
            batch_loss, count = ground_batch(model, batch)
            #print(count)
            train_count += count / batch_size

            batch_loss= batch_loss / batch_size # normalize across the whole batch

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            torch.save(model.state_dict(), "checkpoints/namomo.ckpt")
            train_loss += float(batch_loss)

        if testloader: # just checking if the testing scenario actually exists
            test_loss = 0.0
            test_count = 0
            for batch in testloader:
                batch_loss, count = ground_batch(model, batch)
                batch_loss = batch_loss / batch_size
                test_loss += float(batch_loss)
                test_count += count / batch_size

            writer.add_scalar("test_loss", test_loss / len(testloader), epoch)
            writer.add_scalar("test_percent", test_count/len(testloader) , epoch)
        writer.add_scalar("train_loss", train_loss / len(trainloader), epoch)
        writer.add_scalar("train_percent", train_count /len(trainloader), epoch)
       

        if not (epoch % ckpt_epochs): main_logger.info(f"At Epoch:{epoch}")
    torch.save(model.state_dict(), "checkpoints/namomo.ckpt")
    writer.close()
    return model

def process_command(command):
    if regex.match("train_ccsp_*", command):
        domain = command[11:]
        exec(f"from domains.{domain}.{domain}_domain import {domain}_executor")
        exec(f"from domains.{domain}.{domain}_data import get_constraint_dataset")
        target_executor = eval(f"{domain}_executor")

        main_logger.info(f"start the {command}. ")
        main_logger.info(f"{target_executor.domain.get_summary()}")

        """process the domain dataset"""
        from datasets.ccsp_dataset import collate_graph_batch, Swissroll
        from torch.utils.data import DataLoader
        from core.spatial.energy_graph import TimeInputEnergyMLP, PointEnergyMLP
        from core.spatial.diffusion import ScheduleLogLinear, training_loop

        dataset  = eval("get_constraint_dataset()")
        dataset  = Swissroll(np.pi/2, 5*np.pi, 100)
        loader   = DataLoader(dataset, batch_size=2048, collate_fn=collate_graph_batch)
        constraints = target_executor.constraints
        constraints = {"online" : 1}

        model    = PointEnergyMLP(constraints)
        schedule = ScheduleLogLinear(N=500, sigma_min=0.005, sigma_max=10)
        trainer  = training_loop(loader, model, schedule, epochs=config.epochs)
        losses   = [ns.loss.item() for ns in trainer]
        torch.save(model.state_dict(),"checkpoints/{command}_state.pth")
        main_logger.info("finish the training and saved in checkpoints/{command}_state.pth")

    if regex.match("train_mcl_*", command):
        main_logger.info("start training the metaphorical concept learner")
        domain = command[10:]
        from core.model import EnsembleModel
        from core.metaphors.diagram_legacy import MetaphorMorphism
        model = EnsembleModel(config)
        from domains.generic.generic_domain import generic_executor

        model.concept_diagram.root_name = "Generic"
        model.concept_diagram.add_domain("Generic", generic_executor)
        model.concept_diagram.add_domain("Contact", contact_executor)
        
        for source, target in morphisms:
            model.concept_diagram.add_morphism(source, target, MetaphorMorphism(domains[source], domains[target]))

        text = "this is a real sentence"
        #print(model.encode_text(text).shape) torch.Size([1, 7, 256])
        train_dataset = SceneDataset("contact_experiment", "train")
        test_dataset = SceneDataset("contact_experiment",   "test")
        model = train_grounding(config, model, train_dataset, test_dataset)
    
    if command == "interact":
        main_logger.info("start the interactive mode of the metaphorical concept learner")
        from core.model import EnsembleModel
        model = EnsembleModel(config)
        
        # Import and start the server
        from experiments.server import make_app
        import tornado
        
        # Create and start the server with the model
        app = make_app(model=model)
        port = getattr(config, 'port', 8888)  # Use config port if available, else default to 8888
        app.listen(port)
        
        main_logger.info(f"Server running on http://localhost:{port}")
        
        # Start the Tornado IO loop
        try:
            tornado.ioloop.IOLoop.current().start()
        except KeyboardInterrupt:
            main_logger.info("Server stopped by user")
            tornado.ioloop.IOLoop.current().stop()

    if command == "learn_curriculum":
        main_logger.warning("Curriculum Learning in Progress do not learn")
        from core.model import EnsembleModel, curriculum_learning
        from core.curriculum import load_curriculum
        from domains.generic.generic_domain import generic_executor
        model = EnsembleModel(config)

        if config.load_ckpt:
            #model = torch.load(f"{config.ckpt_dir}/{config.load_ckpt}")
            model = load_ensemble_model(config, f"{config.ckpt_dir}/{config.load_ckpt}")
            main_logger.info(f"loaded checkpoint {config.load_ckpt}")
        else:
            model.concept_diagram.add_domain("Generic", generic_executor)
            model.concept_diagram.root_name = "Generic"

        """load the core knowledge from defined domains"""
        core_knowledge = eval(config.core_knowledge)

        curriculum = load_curriculum(config.curriculum_file) # load the curriculum learning setup

        curriculum_learning(config, model, curriculum) # start the curriculum learning for each block

    return command

if __name__ == "__main__":
    from config import config
    sys.stdout.write(f"command type: {config.command}\n")

    process_command(config.command)