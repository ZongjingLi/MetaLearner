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
    
    circle   (x : List[Object]) : List[Object] := by pass
    square   (x : List[Object]) : List[Object] := by pass
    triangle (x : List[Object]) : List[Object] := by pass
)
"""



objects_domain = load_domain_string(objects_domain_str)

class ObjEncoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.lenet = nn.Sequential(           # input is 1x32x32
            nn.Conv2d(3, 32, 5), nn.ReLU(),   # conv1: 32x28x28
            nn.MaxPool2d(2, 2),               # pool1: 32x14x14
            nn.Conv2d(32, 32, 5), nn.ReLU(),  # conv2: 32x10x10
            nn.MaxPool2d(2, 2),               # pool2: 32x5x5
        )
        self.fc = nn.Sequential(nn.Linear(32 * 5 * 5, out_dim), nn.ReLU())
        self.position_embedding = nn.Embedding(9, 64)
        self.fc2 = nn.Linear(32 * 5 * 5, out_dim)
    
    def forward(self, img):
        obj = self.forward_object(img)
        #rel = self.forward_relation(obj)
        return obj#, rel  # obj: [B, 3, 64], rel: [B, 3, 3, 64]

    def forward_object(self, img):
        # img = B, 3, 30, 30

        b = img.size(0)
        #img = img.reshape((b, 3, 1, 32, 3, 32)).permute((0, 2, 4, 1, 3, 5)).reshape((b * 3, 3, 32, 32))
        img = self.lenet(img)

        img = img.reshape((b, 5 * 5 *32))
        img = self.fc(img)
        return img

    def forward_relation(self, object_feat):
        nr_objects = 3
        position_feature = torch.arange(nr_objects, dtype=torch.int64, device=object_feat.device)
        position_feature = self.position_embedding(position_feature)
        position_feature = position_feature.unsqueeze(0).expand(object_feat.size(0), nr_objects, 64)
        feature = torch.cat([object_feat, position_feature], dim=-1)

        feature1 = feature.unsqueeze(1).expand(feature.size(0), nr_objects, nr_objects, 128)
        feature2 = feature.unsqueeze(2).expand(feature.size(0), nr_objects, nr_objects, 128)
        feature = torch.cat([feature1, feature2], dim=-1)

        feature = self.fc2(feature)
        return feature

class ObjectsExecutor(CentralExecutor):

    def __init__(self, domain):
        super().__init__(domain)
        obj_dim = 32
        self.red_mlp    = nn.Linear(obj_dim, 1)
        self.green_mlp  = nn.Linear(obj_dim, 1)
        self.blue_mlp   = nn.Linear(obj_dim, 1)

        self.circle_mlp    = nn.Linear(obj_dim, 1)
        self.rectangle_mlp = nn.Linear(obj_dim, 1)
        self.square_mlp    = nn.Linear(obj_dim, 1)

        self.object_encoder = ObjEncoder(obj_dim)

    def scene(self):
        if self.grounding is None: self._grounding = {"image":torch.zeros([1,32,32 * 3,3])}
        images = self.grounding["image"]
        sub_images = torch.chunk(images, 3, dim=2)
        encoded_vectors = []
    
        for img in sub_images:

            flat_img = img[None,...]#.reshape([1,-1])

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
        red_logits = self.red_mlp(objects[:,1:]).reshape([-1,1])
        green_logits = self.green_mlp(objects[:,1:]).reshape([-1,1])
        blue_logits = self.blue_mlp(objects[:,1:]).reshape([-1,1])
        color_logits = torch.cat([red_logits, green_logits, blue_logits], dim = -1)

        color_logits = torch.logit(torch.softmax(color_logits, dim = 1))
        
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
    
    def shape_logits(self, objects):
        circle_logits = self.circle_mlp(objects[:,1:]).reshape([-1,1])
        rectangle_logits = self.rectangle_mlp(objects[:,1:]).reshape([-1,1])
        square_logits = self.square_mlp(objects[:,1:]).reshape([-1,1])
        
        shape_logits = torch.cat([circle_logits, rectangle_logits, square_logits], dim = -1)
        shape_logits = torch.logit(torch.softmax(shape_logits, dim = 1))
        
        return shape_logits

    def circle(self, objects):
        circle_logits = self.shape_logits(objects)[:, 0]  # 取圆形的logits
        return torch.cat([
            torch.min(circle_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = -1)

    def rectangle(self, objects):
        rectangle_logits = self.shape_logits(objects)[:, 1]  # 取矩形的logits
        return torch.cat([
            torch.min(rectangle_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = -1)

    def square(self, objects):
        square_logits = self.shape_logits(objects)[:, 2]  # 取正方形的logits
        return torch.cat([
            torch.min(square_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = -1)

objects_executor = ObjectsExecutor(objects_domain)
