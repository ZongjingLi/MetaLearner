# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
import torch.nn as nn
import torch.optim.optimizer

"""create some test domains """
from rinarak.knowledge.executor import CentralExecutor
from rinarak.domain import load_domain_string, Domain
from domains.utils import domain_parser
import sys

lexicon_dim = 256

function_domain_str = """
(domain Function)
(:type
    function - vector[float, 128] ;; abstract repr of function
    set - vector[float, 128] ;; necessary encoding for a state
    point -vector[float, 64] ;; a point in a set
)
(:predicate
    domain ?x-function -> set
    codomain ?x-function -> set
    range ?x-function -> set
    map ?x-function ?y-point -> point ;; how to represent that, why that repr is good
)

"""
function_domain = load_domain_string(function_domain_str, domain_parser)
function_executor = CentralExecutor(function_domain, "cone", lexicon_dim)

objects_domain_str = """
(domain Objects)
(:type
    objects - vector[float, 256] ;; abstract repr of function
)
(:predicate
    feature ?x-objects -> object
)

"""
objects_domain = load_domain_string(objects_domain_str, domain_parser)
objects_executor = CentralExecutor(objects_domain, "cone", lexicon_dim)

path_domain_str = """
(domain Path)
(:type
    path - vector[float, 128] ;; abstract repr of path
    point -vector[float, 64]
)
(:predicate
    start ?x-path -> point
    end ?x-path -> point
    go ?x-point ?y-path -> point
)

"""
path_domain = load_domain_string(path_domain_str, domain_parser)
path_executor = CentralExecutor(path_domain, "cone", lexicon_dim)

set_domain_str = """
(domain Set)
(:type
    set - vector[float, 128]
    point - vector[float,64]
)
(:predicate 
    union ?x-set ?y-set -> set
    intersect ?x-set ?y-set -> set
    compl ?x-set -> set
    subset ?x-set ?y-set -> boolean
    in ?x-point ?y-set -> bool
)
"""
set_domain = load_domain_string(set_domain_str, domain_parser)
set_executor = CentralExecutor(set_domain, "cone", lexicon_dim)

group_domain_str = """
(domain Group)
(:type
    group - vector[float, 72]
    set - vector[float, 128]
    point - vector[float,64]
)
(:predicate 
    mul ?x-point ?y-point -> point
    inv ?x-point ?y-point -> point
    id -> point
)
"""
group_domain = load_domain_string(group_domain_str, domain_parser)
group_executor = CentralExecutor(group_domain, "cone", lexicon_dim)

complex_domain_str = """
(domain Complex)
(:type
    set - vector[float, 128]
    complex - vector[float, 5]
    real - vector[float, 1]
)
(:predicate
    real ?x-complex -> num
    im   ?x-complex -> num
    mul ?x-complex ?y-complex -> complex
    add ?x-complex ?y-complex -> complex
)
"""


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
contact_dom = load_domain_string(domain_str, domain_parser)
contact_executor = CentralExecutor(contact_dom, "cone", 128)

color_domain_str = """
(domain Color)
(:type
    color - vector[float, 3]
)
(:predicate
    red ?x-color -> boolean
    blue ?x-color -> boolean
    green ?x-color -> boolean
    
)
"""
color_domain = load_domain_string(color_domain_str, domain_parser)
color_executor = CentralExecutor(color_domain, "cone", 128)


from domains.distance.distance_domain import distance_executor
from domains.direction.direction_domain import direction_executor
from domains.rcc8.rcc8_domain import rcc8_executor
from domains.logic.logic_domain import build_logic_executor
logic_executor = build_logic_executor()

from core.metaphors.base import *
from core.metaphors.diagram import *


"""1. Create the test Concept Diagram"""
concept_diagram = ConceptDiagram()

nodes = {
	"Objects" : objects_executor,
	"Function" : function_executor,
	"RCC8" : rcc8_executor,
    "Set" : set_executor,
    "Logic" : logic_executor,
    "Contact" : contact_executor,
    "Distance" : distance_executor,
    "Direction" : direction_executor,
    "Color" : color_executor,
}
edges = [
	("Objects", "Function"),
	("Objects", "Set"),
    ("Objects", "Set"),
    ("Objects", "RCC8"),
    ("Objects", "Distance"),
    ("Objects", "Direction"),
    ("Objects", "Color"),
    ("Objects", "Contact"),
	("Function", "Set"),
    #("Set", "RCC8"),
    ("RCC8", "Set"),
    ("Objects", "Logic"),
    ("RCC8", "Contact"),
]

for domain in nodes: concept_diagram.add_node(domain, nodes[domain])
for morph in edges: concept_diagram.add_edge(morph[0], morph[1],)

"""2. Get the lexicon entries in the Concept Diagram"""
key = torch.randn([128])
from rinarak.knowledge.grammar import match_entries, parse_sentence
entries = concept_diagram.get_lexicon_entries()
distrib = match_entries(key, entries)

"""3. Get the path metaphors in the Concept Diagram"""
morphs = concept_diagram.get_edges_between("Objects", "Logic")


"""4. Evaluate predicate along the node"""
schema = Schema(
    end = torch.randn([5, 256]),
    mask  = torch.tensor([1., 0., 1., 1., 1.]),
    dtype = "objects",
    domain = "Objects"
)
predicate = "subset"
input_schemas = [schema, schema]
paths = concept_diagram.get_schema_path(schema, "Contact", "state")
for path in paths:
    print(path[0], path[2])

output_schema = concept_diagram.evaluate_predicate(predicate, input_schemas)


predicate = "partial_overlap"
#predicate = "contact"
input_schemas = [schema, schema]
output_schema = concept_diagram.evaluate_predicate(predicate, input_schemas)
#print(output_schema.end)

"""5. Dataset and Pipeline for Grounding"""
class GroundingDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 0
    
    def __getitem__(self):
        """the return data should be a diction withe the following values
        inputs : a single input scene
        domain : the domain for the encoder to extract features
        program : the program structure for the execution
        answer : the answer for the grounding backend
        query : optional, if can translate the query to a program
        """
        data = {}
        return data

from tqdm import tqdm
from datasets.sprite_contact_dataset import SpritesContactDataset

dataset = SpritesContactDataset(level = 1)

def list_collate_fn(batch):
    """
    Custom collate function that keeps individual items in a list without stacking/batching.
    This preserves the original dictionary structure of each item.
    
    Args:
        batch (list): List of samples from the dataset, each being a dictionary
        
    Returns:
        list: A list of dictionaries, each dictionary representing one sample
    """
    # Simply return the list of dictionaries without any processing
    return batch
# when collate the dataset in the dataloader, don't make it batch wise, just use individual scenes

history = []
import matplotlib.pyplot as plt
def train(model, dataset, epochs = 50):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 50, shuffle = True,
                    collate_fn=list_collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
    for epoch in range(epochs):
        for sample_list in train_loader:#dataset:
            loss = 0.0
            optimizer.zero_grad()
            #target_predicate = "north"
            for target_predicate in ["north"]:#, "south", "west", "east"]:
                for sample in sample_list:
                    #ground_gt = sample["contact_matrix"]
                    ground_gt = sample["direction_matrices"][target_predicate]
                
                    masks = sample["masks"].squeeze(1).unsqueeze(0)
                    image = sample["image"].unsqueeze(0)
                    n_obj = masks.shape[1]

                    objects_encoded = model.encoders["image"](image, masks)[0]

                    input_schema = Schema(
                        end = objects_encoded,
                        mask = torch.ones([n_obj]) - 1e-6,
                        dtype = "objects", domain = "Objects")
                    input_schemas = [input_schema, input_schema]
                    predicate = f"{target_predicate}"
                    output_schema = model.concept_diagram.evaluate_predicate(predicate, input_schemas)

                    ground_predict = output_schema.end
                    mask_predict = output_schema.mask

                    loss += torch.nn.functional.binary_cross_entropy_with_logits(ground_predict, ground_gt, reduction = "mean")
                    #loss += torch.nn.functional.binary_cross_entropy( mask_predict, torch.ones_like(mask_predict), reduction = "mean")
            loss = loss / len(sample_list)
            loss.backward()
            optimizer.step()
            history.append(loss.detach())

            paths = concept_diagram.get_schema_path(schema, "Contact", "state")
            for path in paths:
                print(path[0], path[2])

            sys.stdout.write(f"\repoch:{epoch} loss:{loss}")
            plt.plot(history)
            plt.pause(0.0001)
            plt.cla()
    #print(ground_predict)
    #print(ground_gt)

    return model
            
        

from config import config
from core.model import EnsembleModel
model = EnsembleModel(config)
model.concept_diagram = concept_diagram

model = train(model, dataset, epochs = 15)

for sample in dataset:
    plt.subplot(121)
    masks = sample["masks"].squeeze(1).unsqueeze(0)
    image = sample["image"].unsqueeze(0)
    n_obj = masks.shape[1]

    objects_encoded = model.encoders["image"](image, masks)[0]

    input_schema = Schema(
    end = objects_encoded,
    mask = torch.ones([n_obj]) - 1e-6,
        dtype = "objects", domain = "Objects")
    #input_schemas = [input_schema, input_schema]
    plt.imshow(image[0].permute(1,2,0))

    paths = model.concept_diagram.get_schema_path(input_schema, "Distance", "state")
    for path in paths:
        print(path[0], path[2])
    
    plt.subplot(122)
    #plt.xlim(-1, 1.0)
    #plt.ylim(-1, 1.0)
    coords = path[1][-1].end.detach()

    coords_x, coords_y = coords[:,0], coords[:,1]
    plt.scatter(coords_x, coords_y)

    plt.show()