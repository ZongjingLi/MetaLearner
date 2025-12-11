'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-12-10 13:28:04
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-10 13:28:20
'''
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string, Domain
from helchriss.dsl.dsl_types import EmbeddingType
from helchriss.utils import stprint

memory_domain_str = """
(domain :: Memory)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    key -  Embedding[key, 128]
    value  - Embedding[value, 128]
)
(def function
    query (x : key) : Tuple[boolean, value] := by pass
    write (x : key) (y : value) : boolean := by pass
    recall (x : key)  : boolean := by pass
    clear : boolean := by pass
)
"""


memory_domain = load_domain_string(memory_domain_str)


class MemoryExecutor(CentralExecutor):

    def __init__(self, domain : Domain):
        super().__init__(domain)
        self.key_type = domain.type_aliases["key"][1]
        self.value_type = domain.type_aliases["value"][1]

        assert isinstance(self.key_type, EmbeddingType), f"{self.key_type} is not a Embedding Type"
        assert isinstance(self.value_type, EmbeddingType), f"{self.value_type} is not a Embedding Type"
        
        self.key_dim = int(self.key_type.dim)
        self.value_dim = int(self.value_type.dim)


        self.key_store = nn.Parameter(torch.empty(0, self.key_dim), requires_grad=True)
        self.value_store = nn.Parameter(torch.empty(0, self.value_dim), requires_grad=True)
        self.max_entries = 100
        self.temperature = 0.1

    def query(self, x):
        if self.key_store.numel() == 0:
            return [(torch.tensor(0.0, requires_grad=True), 
                     torch.zeros(self.value_dim, requires_grad=True))]

        x_norm           =  x / torch.norm(x, p=2, keepdim=True)
        key_store_norm   =  self.key_store / torch.norm(self.key_store, p=2, dim=1, keepdim=True)
        
        similarities     =  torch.matmul(key_store_norm, x_norm.unsqueeze(-1)).squeeze(-1)  # [num_entries]
        weights          =  torch.softmax(similarities / self.temperature, dim=0)  # [num_entries], sums to 1
        
        expected_value   =  torch.matmul(weights.unsqueeze(0), self.value_store).squeeze(0)  # [value_dim]
        

        match_confidence =  torch.logit(torch.max(torch.sigmoid(10 * (similarities - 0.5))) )
        
        return [(match_confidence, expected_value)]

    def write(self, x, y):
        dim_check = x.shape == (self.key_dim,) and y.shape == (self.value_dim,)

        assert dim_check, f"{x.shape} not eq {self.key_dim} or {y.shape} not eq {self.value_dim}"#: return torch.tensor(0.0, requires_grad=True)
        
        if self.key_store.shape[0] >= self.max_entries: return torch.tensor(0.0, requires_grad=True)
        
        self.key_store = nn.Parameter(
            torch.cat([self.key_store, x.unsqueeze(0)], dim=0),requires_grad=True)
        self.value_store = nn.Parameter(
            torch.cat([self.value_store, y.unsqueeze(0)], dim=0),requires_grad=True)
        
        return torch.tensor(1.0, requires_grad=True)


    def clear(self) -> torch.Tensor:
        # reinitialize stores with empty tensors (preserve parameter properties)
        self.key_store = nn.Parameter(torch.empty(0, self.key_dim), requires_grad=True)
        self.value_store = nn.Parameter(torch.empty(0, self.value_dim), requires_grad=True)
        return torch.tensor(1.0, requires_grad=True)


memory_executor = MemoryExecutor(memory_domain)


if __name__ == "__main__":
    dim = 128
    key1 = torch.randn([dim])
    val1 = torch.ones([dim])

    key2 = torch.randn([dim])
    val2 = torch.ones([dim])

    memory_executor.write(key1, val1)
    memory_executor.write(key2, val2)


    print(memory_executor.query(torch.randn([dim])))