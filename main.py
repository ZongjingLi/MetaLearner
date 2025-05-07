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
import yaml
import torch
import numpy as np
from tqdm import tqdm
from helchriss.logger import get_logger, set_output_file


main_logger = get_logger("Main")
set_output_file("outputs/logs/main_logs.txt")

def create_prototype_metalearner(config):
    core_knowledge_config = config.core_knowledge
    checkpoint_dir = config.load_model
    main_logger.info(f"using the core knowledge stored in {core_knowledge_config} to create prototype")
    if checkpoint_dir:
        main_logger.info(f"loading the checkpoint parameters in {checkpoint_dir}")
    else: main_logger.warning("not loading any checkpoint!!!")
    with open(core_knowledge_config, 'r') as file:
        model_config = yaml.safe_load(file)

    """start to load the model config, core knowledge and lexicon associated with it"""
    model_name = model_config["name"]

    # load the domain functions used for the meta-learner
    domain_executors = []
    domain_infos = {}
    for domain_name in model_config["domains"]:
        domain =  model_config["domains"][domain_name]
        path = domain["path"]
        name = domain["name"]
        domain_infos[domain_name] = {"path" : path, "name" : name}
        exec(f"from {path} import {name}")
        domain_executors.append(eval(name))

    # load the lexicon entries and the vocab learned in the meta-learner
    vocab_path = model_config["vocab"]["path"]

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = [line.strip() for line in f]

    from core.model import MetaLearner
    model = MetaLearner(domain_executors, vocab)
    model.domain_infos = domain_infos
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

    if regex.match("train_mcl", command):
        main_logger.info("start training the metaphorical concept learner")
        from core.model import MetaLearner

        with open(config.dataset_config, 'r') as file:
            dataset_config = yaml.safe_load(file)

        load_name = config.load_model
        model = MetaLearner([], [])
        model = model.load_ckpt(f"outputs/checkpoints/{load_name}")

        dataset_name     = config.dataset_name
        train_epochs     = config.epochs
        train_lr         = config.lr
        dataset_path     = dataset_config[dataset_name]["path"]
        dataset_getter   = dataset_config[dataset_name]["getter"]
        exec(f"from {dataset_path} import {dataset_getter}")
        train_dataset = eval(f"{dataset_getter}()")

        model.train(train_dataset, epochs = train_epochs, lr = train_lr)

        save_name = config.save_model
        model.save_ckpt(f"outputs/checkpoints/{save_name}")



    if regex.match("interact_*", command):
        import tornado
        from assets.app import make_app
        from core.model import MetaLearner
        model_name = command[9:]
        main_logger.info(f"try to interact with model {model_name}.")


        model = MetaLearner([], [])
        model = model.load_ckpt(f"outputs/checkpoints/{model_name}")

        """start the interaction with the model using the web application"""
        app = make_app(model)
        os.system("lsof -ti:8888 | xargs kill -9")
        app.listen(8888)
        main_logger.info("server started at http://localhost:8888")
        tornado.ioloop.IOLoop.current().start()

    if regex.match("create_*", command):
        model_name = command[7:]
        model = create_prototype_metalearner(config)
        model.entries_setup()
        
        from helchriss.knowledge.symbolic import Expression
        expr = Expression.parse_program_string("smaller:Integers(inf:Order(), sup:Order())")
        model.infer_metaphor_expressions([expr])

        model.save_ckpt("outputs/checkpoints/prototype")
        main_logger.info(f"created model {model_name} and saved successfully.")



if __name__ == "__main__":
    from config import config

    sys.stdout.write(f"command type: {config.command}\n")

    process_command(config.command)

    