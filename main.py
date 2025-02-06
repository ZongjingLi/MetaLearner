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

main_logger = get_logger("Main")
set_output_file("logs/main_logs.txt")

def train_grounding(model, scene_dataset : 'SceneDataset', config):
    epochs = int(config.epochs)
    batch_size = int(config.batch_size)
    ckpt_epochs = int(config.ckpt_epochs)
    loader  = DataLoader(scene_dataset, batch_size, collate_fn = scene_collate,shuffle = True)
    for epoch in tqdm(range(epochs)):
        for batch in loader:
            batch_loss = 0.0
            batch_size = len(batch["input"])
            for i,scene in enumerate(batch["input"]): # this is dump but some kind of batchwise operation
                scene = torch.stack(scene) # normally a nx... input scene
                #print(scene.shape)
                for pred in batch["predicate"]:
                    if pred == "end": break
                    pred # the name fo the predicate
                    batch["predicate"][pred][0][i] # tensor repr of the predicate

                    results = model.evaluate(scene, pred, encoder_name = "pointcloud")
                    batch_loss += 0.0

            batch_loss = batch_loss / batch_size # normalize across the whole batch

        if not (epoch % ckpt_epochs): main_logger.info(f"At Epoch:{epoch}")
    torch.save(model, "checkpoints/namomo.ckpt")
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
        model = EnsembleModel(config)

        text = "this is a real sentence"
        #print(model.encode_text(text).shape) torch.Size([1, 7, 256])
        dataset = SceneDataset("contact_experiment", "train")
        model = train_grounding(model, dataset, config)
    
    if command == "interact":
        main_logger.info("start the interactive mode of the metaphorical concept learner")
        from core.model import EnsembleModel
        model = EnsembleModel(config)
        
        # Import and start the server
        from server import make_app
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
        model = EnsembleModel(config)
        curriculum_learning(model, [])

    return command

if __name__ == "__main__":
    from config import config
    sys.stdout.write(f"command type: {config.command}\n")

    process_command(config.command)