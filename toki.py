from core.metaphors.types import *
from core.metaphors.diagram_executor import *

tp1 = TypeSpaceBase.parse_type("boolean")
tp2 = TypeSpaceBase.parse_type("vector[float,[32]]")
#tp3 = TypeSpaceBase.parse_type("List[boolean]")

caster = TypeCaster(tp1, tp2)

v1 = torch.randn([6,1])

v2, p = caster(v1)

print(v2.shape)
print(p)

from core.model import Aluneth, SceneGroundingDataset
from domains.numbers.integers_domain import integers_executor
domains = [
    integers_executor, #objects_executor
]
executor = MetaphorExecutor(domains)

