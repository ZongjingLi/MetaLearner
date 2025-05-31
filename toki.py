
from helchriss.knowledge.symbolic import Expression
from datasets.mathprim.numbers_dataset import get_dataset
from core.model import MetaLearner
from domains.spatial.direction_domain import direction_executor
from domains.math.integer_domain import integers_executor

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
#model.parser.purge_entry(word, 0.01, abs = 0)
model.parser.display_word_entries(word)

query = "one plus two plus three"

results = model.parser.parse(query, forced = True)

parse = str(model.maximal_parse(query)[0][0].sem_program)
print(parse)
print(model.parser.generate_sentences_for_program(parse))

outputs = model.forward(query, {})
model.executor.display()


#print(outputs[0])
print(len(outputs[0]))
