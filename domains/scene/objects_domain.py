'''
 # @Author: Yiqi Sun
 # @Create Time: 2025-11-30 23:47:48
 # @Modified by: Yiqi Sun
 # @Modified time: 2025-12-10 13:28:32
'''
import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from helchriss.utils import stprint

objects_domain_str = """
(domain :: Objects)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    Object -  Embedding[object, 96]
    Shape  - Embedding[shape, 32]
    Color - Embedding[color, 3]
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    scene : List[Tuple[boolean,Object]] := by pass

    color (x : Object) : Color := by pass
    shape (x : Object) : Shape := by pass

    red        (x : Object) : boolean := by pass
    green      (x : Object) : boolean := by pass
    blue       (x : Object) : boolean := by pass
    
    circle     (x : Object) : boolean := by pass
    rectangle  (x : Object) : boolean := by pass
    triangle   (x : Object) : boolean := by pass

    left  (x y : Object): boolean := by pass
    right (x y : Object): boolean := by pass
)
"""

def stable_softmax(logits, dim=-1, eps=1e-6):
    return torch.softmax(logits, dim = dim)
    logits = logits - torch.max(logits, dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    probs = exp_logits / (torch.sum(exp_logits, dim=dim, keepdim=True) + eps)
    return probs

objects_domain = load_domain_string(objects_domain_str)

class CNNObjEncoder(nn.Module):
    def __init__(self, output_dim=64, spatial_dim = 32):
        super().__init__()
        self.lenet = nn.Sequential(           # input is 1x32x32
            nn.Conv2d(3, 32, 5), nn.ReLU(),   # conv1: 32x28x28
            nn.MaxPool2d(2, 2),               # pool1: 32x14x14
            nn.Conv2d(32, 32, 5), nn.ReLU(),  # conv2: 32x10x10
            nn.MaxPool2d(2, 2),               # pool2: 32x5x5
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU(),
            )
        self.position_embedding = nn.Embedding(9, spatial_dim)
        self.fc2 = nn.Linear(32 * 5 * 5, output_dim)
    
    def forward(self, img):
        obj = self.forward_object(img)
        #rel = self.forward_relation(obj)
        return obj#, rel  # obj: [B, 3, 64], rel: [B, 3, 3, 64]

    def forward_object(self, img):
        # img = B, 3, 30, 30

        b = img.size(0)
        #img = img.reshape((b, 3, 1, 32, 3, 32)).permute((0, 2, 4, 1, 3, 5)).reshape((b * 3, 3, 32, 32))
        img = self.lenet(img)


        img = img.reshape((b, -1))
        img = self.fc(img)

        positions = torch.arange(b, device=img.device)
        #print(positions)

        pos_embed = self.position_embedding(positions)
        combined = torch.cat([img, pos_embed], dim=1)
        return combined
    
class FCBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=1, activation=nn.ReLU(), dropout=0.1):
        super().__init__()
        
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        #layers.append(nn.Dropout(dropout))
    
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
            #layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x): return self.net(x)

class ObjectsExecutor(CentralExecutor):

    def __init__(self, domain):
        super().__init__(domain)
        
        spatial_dim = 32
        feature_dim = 64
        obj_dim = feature_dim + spatial_dim
        self.red_mlp    = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)
        self.green_mlp  = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)
        self.blue_mlp   = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)

        self.circle_mlp    = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)
        self.rectangle_mlp = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)
        self.triangle_mlp  = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)

        self.left_mlp  = FCBlock(obj_dim + obj_dim, 1)
        self.right_mlp = FCBlock(obj_dim + obj_dim, 1)

        self.object_encoder = CNNObjEncoder(output_dim = feature_dim, spatial_dim = spatial_dim)
        #self.object_encoder = MLPObjEncoder(output_dim = obj_dim)#
        self.device = "cpu"#mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def scene(self, **kwargs):
        device = self.device

        if "image" not in self.grounding:
            self._grounding = {"image":torch.zeros([3,32,32 * 3])}
            print("warning: no image provided")
        images = self.grounding["image"]
        sub_images = torch.chunk(images, 3, dim=2)

        result = self.object_encoder(torch.stack(sub_images))

        result = torch.cat([
            torch.ones([3, 1], device = device) * 13,
            result
        ], dim = 1)
        
        return result

    def color_logits(self, objects):
        red_logits     = self.red_mlp(objects)
        green_logits   = self.green_mlp(objects)
        blue_logits    = self.blue_mlp(objects)
        color_logits = torch.cat([red_logits, green_logits, blue_logits], dim = -1)
        return color_logits

    def red(self, objects):        
        return self.color_logits(objects)[:,0]

    def green(self, objects):
        return self.color_logits(objects)[:,1]

    def blue(self, objects):
        return self.color_logits(objects)[:,2]

    def shape_logits(self, objects):
        circle_logits    = self.circle_mlp(objects)
        rectangle_logits = self.rectangle_mlp(objects)
        triangle         = self.triangle_mlp(objects)
        
        shape_logits = torch.cat([circle_logits, rectangle_logits, triangle], dim = 1)
        #shape_logits = torch.logit(stable_softmax(shape_logits, dim = 1), eps = 1e-6)
        return shape_logits

    def circle(self, objects):
        return self.shape_logits(objects)[:, 0]

    def rectangle(self, objects):
        return  self.shape_logits(objects)[:, 1]

    def triangle(self, objects):
        return self.shape_logits(objects)[:, 2]

    
    def relation_features(self, anchor_object, ref_objects):
        anchor_features = anchor_object[:,1:] # [nxd]
        ref_features = ref_objects[:, 1:] # [nxd]

        anchor_expanded = anchor_features.unsqueeze(1).expand(-1, ref_features.size(0), -1)
        ref_expanded = ref_features.unsqueeze(0).expand(anchor_features.size(0), -1, -1)
        feature_matrix = torch.cat([anchor_expanded, ref_expanded], dim=2)

        return feature_matrix

    def left(self, anchor_object, ref_objects):
        return self.left_mlp(torch.cat([anchor_object, ref_objects], dim = -1))
    
    def right(self, anchor_object, ref_objects):
        return self.right_mlp(torch.cat([anchor_object, ref_objects], dim = -1))

objects_executor = ObjectsExecutor(objects_domain)
