# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
import torch
from domains.scene.objects_domain import objects_executor
from domains.scene.slice_scene_domain import slice_scene_executor
from domains.numbers.integers_domain import integers_executor
from domains.logic.fol_domain import fol_executor
from domains.visual.color_domain import color_executor
from domains.structure.order_domain import order_executor
from domains.spatial.circle_domain import circle_executor


from core.model import Aluneth, SceneGroundingDataset
from helchriss.dsl.dsl_values import Value
from typing import List


domains = [
    integers_executor,
    color_executor,
    #order_executor,
    #circle_executor,
    slice_scene_executor
    #objects_executor
]

from datasets.marked_integer_dataset import MixedSprites3Dataset


import numpy as np

def formal_answer(answer):
    # Normalize strings
    if isinstance(answer, str):
        lower = answer.strip().lower()
        if lower in ("true", "1", "yes"):
            return Value("boolean", 1.0)
        if lower in ("false", "0", "no"):
            return Value("boolean", 0.0)
        try:
            if "." in lower:
                return Value("float", float(lower))
            else:
                return Value("int", int(lower))
        except ValueError:
            raise ValueError(f"Cannot parse string input: {answer}")

    # Handle booleans
    if isinstance(answer, bool):
        return Value("boolean", float(answer))

    # Handle NumPy booleans
    if isinstance(answer, (np.bool_,)):
        return Value("boolean", float(bool(answer)))

    # Handle integers (including NumPy int types)
    if isinstance(answer, (int, np.integer)):
        return Value("int", int(answer))

    # Handle floats (including NumPy float types)
    if isinstance(answer, (float, np.floating)):
        return Value("float", float(answer))

    raise TypeError(f"Unsupported type for answer: {type(answer)}")


from datasets.numbers_dataset import get_dataset
num_dataset = get_dataset()

train_dataset = MixedSprites3Dataset(dataset_size=1028)  # create a dataset with 1024 samples
visual_questions = [data["question"] for i,data in train_dataset]
visual_answers = [formal_answer(data["answer"]) for i,data in train_dataset]


visual_groundings = [{"image":data["image"]} for i,data in train_dataset]
visual_dataset = SceneGroundingDataset(visual_questions, visual_answers, visual_groundings)


def gather_grounding_vocab(datasets : List[SceneGroundingDataset]) -> List[str]:
    vocab = []
    for dataset in datasets:
        for idx,data in dataset: vocab.append(data["query"].replace("?",""))
    return vocab


from helchriss.utils.vocab import build_vocab
query_corpus = gather_grounding_vocab([visual_dataset, num_dataset])
vocab = build_vocab(query_corpus)

model = Aluneth(domains, vocab)
model.load_state_dict(torch.load("outputs/checkpoints/model_aluneth.pth"))



"""Phase1 : Learning the sentences parsing in counting domain"""
model.train(num_dataset, epochs = 100, lr = 1e-1, topK = None)


"""Phase2 : Learning the parsing and concepts in the visual domain """
#model.train(visual_dataset, epochs = 1, lr = 1e-1, topK = None)


print("\nLiteral Sancheck\n")
direct_parse = model.parser.parse("red plus two", topK = 2, forced = True)
direct_probs = model.parser.get_parse_probability(direct_parse)

for (parse, logp) in zip(direct_parse, direct_probs):
    print(parse.sem_program, logp.exp())

"""



# Now let's visualize a few samples from the dataset:
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    #print(train_dataset[i])
    plt.imshow(train_dataset[i]['image'].permute(1, 2, 0).numpy()[..., ::-1] * 0.5 + 0.5)
    plt.title(str(train_dataset[i]['question']) + ': ' + str(train_dataset[i]['answer']))
    stprint(train_dataset[i])
    from helchriss.knowledge.symbolic import Expression
    expr = Expression.parse_program_string("scene_objects()")
    embds = slice_scene_executor.evaluate(expr, train_dataset[i])

plt.tight_layout()
#plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(18, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    #print(train_dataset[i])
    plt.imshow(train_dataset[i]['image'].permute(1, 2, 0).numpy()[..., ::-1] * 0.5 + 0.5)
    plt.title(str(train_dataset[i]['question']) + ': ' + str(train_dataset[i]['answer']))
    print(train_dataset[i])
    from helchriss.knowledge.symbolic import Expression
    expr = Expression.parse_program_string("scene_objects()")
    embds = slice_scene_executor.evaluate(expr, train_dataset[i])

plt.tight_layout()
plt.show()

"""

#torch.save(model.state_dict(), "outputs/checkpoints/model_aluneth.pth")
