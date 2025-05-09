
from helchriss.knowledge.symbolic import Expression
from datasets.numbers_dataset import get_dataset
from core.model import MetaLearner
from domains.spatial.direction_domain import direction_executor
from domains.numbers.integers_domain import integers_executor

model_name = "prototype"
model : MetaLearner = MetaLearner([direction_executor, integers_executor], [])
#model = model.load_ckpt(f"outputs/checkpoints/{model_name}")

expr = Expression.parse_program_string("north:Direction(one:Integers(), two:Integers())")
#expr = Expression.parse_program_string("plus:Integers(one:Integers(), two:Integers())")

model.infer_metaphor_expressions(expr)

expr = Expression.parse_program_string("bigger:Integers(one:Integers(), two:Integers())")

model.executor.init_graph()
outputs = model.executor.evaluate(expr, {})
graph = model.eval_graph()
model.executor.display()

print(outputs)

#import networkx as nx
#import matplotlib.pyplot as plt
#nx.draw(graph["graph"])
#plt.show()

