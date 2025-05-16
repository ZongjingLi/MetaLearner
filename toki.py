
from helchriss.knowledge.symbolic import Expression
from datasets.mathprim.numbers_dataset import get_dataset
from core.model import MetaLearner
from domains.spatial.direction_domain import direction_executor
from domains.math.integers_domain import integers_executor

model_name = "prototype"
model : MetaLearner = MetaLearner([direction_executor, integers_executor], [])
#model = model.load_ckpt(f"outputs/checkpoints/{model_name}")

expr = Expression.parse_program_string("north:Direction(one:Integers(), two:Integers())")
#expr = Expression.parse_program_string("plus:Integers(one:Integers(), two:Integers())")

model.infer_metaphor_expressions(expr)

expr = Expression.parse_program_string("bigger:Integers(one:Integers(), two:Integers())")



model_name = "sizer"
word = "bigger"
model = model.load_ckpt(f"outputs/checkpoints/{model_name}")
model.parser.purge_entry(word, 0.01, abs = 0)
model.parser.display_word_entries(word)

query = "one plus two bigger two plus three plus one"


print(model.maximal_parse(query)[0][0])

outputs = model.forward(query, {})
model.executor.display()


#print(outputs[0])
print(len(outputs[0]))