# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
from domains.numbers.integers_domain import integers_executor
from domains.scene.objects_domain import objects_executor
from domains.logic.fol_domain import fol_domain_str

from core.model import Aluneth, SceneGroundingDataset
from helchriss.dsl.dsl_values import Value

domains = [
    integers_executor, #objects_executor
]

from config import config
from helchriss.utils.os import load_corpus
from helchriss.utils.vocab import build_vocab
corpus = load_corpus(config.corpus)
vocab = build_vocab(corpus)


test_sentences = ["two plus one", "two plus three", "one plus one"]
test_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0)]

sum_dataset = SceneGroundingDataset(test_sentences, test_answers, groundings = None)

#vocab = ["one", "plus", "two", "three"]

vocab = build_vocab(test_sentences)
model = Aluneth(domains, vocab)

from helchriss.utils import stprint
stprint(vocab)
stprint(test_answers)

model.train(sum_dataset, epochs = 500, lr = 1e-1)

model.parse_display("one")
model.parse_display("two")
model.parse_display("three")

model.parse_display("<END>")