""" a working typing system"""


""" a union of executor bundles"""
from domains.numbers.integers_domain import integers_executor
from domains.structure.order_domain import order_executor
from domains.visual.color_domain import color_executor
domains = [
    integers_executor, order_executor, color_executor #objects_executor
]


from core.metaphors.diagram_executor import ExecutorGroup, ReductiveUnifier, ReductiveExecutor
base_executor = ExecutorGroup(domains, concept_dim = 128)
executor = ReductiveExecutor(base_executor)


meta_expr = executor.parse_expression("lesser:Order(one:Integers(),two:Integers())")
from helchriss.utils import stprint
infers = executor.infer_reductions(meta_expr)

executor.add_metaphors(infers)

#executor.add_reduction("smaller:Integers", "lesser:Order")



res = executor.evaluate("plus:Integers(one:Integers(), two:Integers())", {})
executor.display()

res = executor.evaluate("lesser:Order(inf:Order(), sup:Order())", {})
executor.display()

res = executor.evaluate("smaller:Integers(inf:Order(), sup:Order())", {})
executor.display()

res = executor.evaluate("lesser:Order(one:Integers(),two:Integers())", {})
executor.display()
print(res)
