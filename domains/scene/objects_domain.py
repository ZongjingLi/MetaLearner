import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string

objects_domain_str = """
(domain :: Objects)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    object - Tuple[boolean, Embedding[object, 64]]
    shape  - Tuple[boolean, Embedding[shape, 32]]
    color - Embedding[color, 3]
    
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    exists (x : List[object]) : boolean := by pass
    forall (x : List[object]) : boolean := by pass
    iota   (x : List[object]) : List[object] := by pass

    negate (x : boolean) : boolean := by pass
    logic_and (x y : boolean) : boolean := by pass
    logic_or  (x y : boolean) : boolean := by pass

    assert {var : Type} (x : var) (y : var -> boolean) : boolean := by pass
    
    count (x : List[object]) : integer := by pass
    scene : List[object] := by pass

    red      (x : List[object]) : List[object] := by pass
    green    (x : List[object]) : List[object] := by pass
    blue     (x : List[object]) : List[object] := by pass
    
    circle   (x : List[shape]) : List[shape] := by pass
    square   (x : List[shape]) : List[shape] := by pass
    triangle (x : List[shape]) : List[shape] := by pass
)
"""



objects_domain = load_domain_string(objects_domain_str)

class ObjEncoder(nn.Module):
    def __init__(self, in_dim = 128, out_dim = 64):
        super().__init__()
        self.linear0 = nn.Linear(in_dim, 128)
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, out_dim)
    
    def forward(self, x):
        x = self.linear0(x)
        x = torch.relu(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

class ObjectsExecutor(CentralExecutor):

    def __init__(self, domain):
        super().__init__(domain)
        obj_dim = 64
        self.red_mlp    = nn.Linear(obj_dim, 1)
        self.green_mlp  = nn.Linear(obj_dim, 1)
        self.blue_mlp   = nn.Linear(obj_dim, 1)

        self.object_encoder = ObjEncoder(32 * 32 * 3, obj_dim)

    def scene(self):
        if self.grounding is None: self._grounding = {"image":torch.zeros([1,32,32 * 3,3])}
        images = self.grounding["image"]
        sub_images = torch.chunk(images, 3, dim=2)
        encoded_vectors = []
    
        for img in sub_images:
            flat_img = img.reshape([1,-1])

            encoded = self.object_encoder(flat_img)
            encoded_vectors.append(encoded)
        result = torch.cat(encoded_vectors, dim=0)
        result = torch.cat([
            torch.ones([3, 1]) * 10,
            result
        ], dim = 1)
        return result

    def exists(self, objects): return torch.max(objects[: ,0])

    def forall(self, objects): return torch.min(objects[:, 0])
    
    def iota(self, objects): return torch.cat([
        torch.logit(torch.softmax(objects[:, 0], dim = 0).reshape([-1,1])),
          objects[:,1:]
    ], dim = -1)

    def negate(self, logit): return -logit

    def logic_and(self, logit1, logit2): return torch.min(logit1, logit2)

    def logic_or(self, logit1, logit2): return torch.max(logit1, logit2)

    def count(self, objects): return torch.sum(torch.sigmoid(objects[:,0]))

    def color_logits(self, objects):
        red_logits = self.red_mlp(objects[:,1:]).reshape([-1])
        green_logits = self.green_mlp(objects[:,1:]).reshape([-1])
        blue_logits = self.blue_mlp(objects[:,1:]).reshape([-1])
        color_logits = torch.cat([red_logits, green_logits, blue_logits], dim = 0)
        color_logits = torch.logit(torch.softmax(color_logits, dim = 0))
        return color_logits

    def red(self, objects):
        red_logits = self.color_logits(objects)[0]

        return torch.cat([torch.min(red_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def green(self, objects):
        green_logits = self.color_logits(objects)[1]

        return torch.cat([torch.min(green_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def blue(self, objects):
        blue_logits = self.color_logits(objects)[2]
        return torch.cat([torch.min(blue_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

objects_executor = ObjectsExecutor(objects_domain)
