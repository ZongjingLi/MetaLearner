# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-19 20:32:58
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-28 12:42:51
from config import config
from core.model import Aluneth, SceneGroundingDataset
from helchriss.dsl.dsl_values import Value

from helchriss.utils.os import load_corpus
from helchriss.utils.vocab import build_vocab


test_sentences = ["two plus one", "two plus three", "one plus one", "red object plus one"]
test_answers = [Value("int",3.0),Value("int",5.0), Value("int", 2.0), Value("int", 3.0)]
sum_dataset = SceneGroundingDataset(test_sentences, test_answers, groundings = None)


from domains.spatial.path_domain import path_executor

domains = [path_executor]

vocab = build_vocab(test_sentences)
model = Aluneth(domains, vocab)



model.train(sum_dataset, epochs = 10, lr = 1e-1)

model.parse_display("one")
model.parse_display("two")
model.parse_display("three")

print("start the literal sancheck of the domains")

model.parser.literal_sancheck("one plus two")

for entry in model.parser.get_likely_entries("one"):print(entry)

