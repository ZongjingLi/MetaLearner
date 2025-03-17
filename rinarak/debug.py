from rinarak.program import *
from rinarak.types import *
from rinarak.dsl.logic_primitives import *

tone = Primitive("1",arrow(tint), 1)
plus = Primitive("plus",arrow(tint,tint,tint), lambda x: lambda y: x + y)

times = Primitive("pow", arrow(tint, tint), lambda x: 2 ** x)

p = Program.parse("(plus 1 1)")
p = Program.parse("(pow 1)")

o = p.evaluate({})

print(o)