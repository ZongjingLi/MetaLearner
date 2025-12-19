import torch
import torch.nn as nn
from helchriss.knowledge.executor import CentralExecutor
from helchriss.domain import load_domain_string
from helchriss.utils import stprint

objects_domain_str = """
(domain :: Blockworld)
(def type  ;; define type alias using a - b, meaning a is an alias to type b
    ;;ground - Embedding[horizon2d, 3] ;; left coordinate, right coordinate, height
    block - Embedding[block2d, 4]
    pos - Vector[float,2] ;; the position of a block
    action - Embedding[action, 4]
)
(def function
    ;; by pass is defaulty used to avoid the actual definion of the functions
    position (x : block) : pos := by pass
    contact (x y : block) : boolean := by pass
    on (x y : block) : boolean := by pass

    floor : block := by pass
)
(def constraint
    on : (x y : block)
    contact : (x y : block)
    v_aligned : (x y : block) ;; vertial aligned
    h_aligned : (x y : block) ;; horizontal aligned
)
(def action
    pick (x : block)
)
"""

def stable_softmax(logits, dim=-1, eps=1e-6):
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
            nn.Linear(32 * 5 * 5, output_dim),
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
        feature_dim = 64
        spatial_dim = 32
        obj_dim = feature_dim + spatial_dim
        self.red_mlp    = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)
        self.green_mlp  = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)
        self.blue_mlp   = FCBlock(obj_dim,1) #nn.Linear(obj_dim, 1)

        self.circle_mlp    = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)
        self.rectangle_mlp = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)
        self.triangle_mlp  = FCBlock(obj_dim,1)#nn.Linear(obj_dim, 1)

        self.left_mlp  = FCBlock(obj_dim + obj_dim, 1)
        self.right_mlp = FCBlock(obj_dim + obj_dim, 1)

        self.object_encoder = CNNObjEncoder(output_dim = feature_dim, spatial_dim = spatial_dim)#MLPObjEncoder(output_dim = obj_dim)#
        self.device = "cpu"#mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    def scene(self):
        device = self.device
        if "image" not in self.grounding: self._grounding = {"image":torch.zeros([3,32,32 * 3])}
        images = self.grounding["image"]
        sub_images = torch.chunk(images, 3, dim=2)
        encoded_vectors = []
    
        for img in sub_images:

            flat_img = img[None,...]#img.reshape([1,-1]) #

            encoded = self.object_encoder(flat_img)
            encoded_vectors.append(encoded)
        result = torch.cat(encoded_vectors, dim=0)
        result = torch.cat([
            torch.ones([3, 1], device = device) * 13,
            result
        ], dim = 1)
        return result

    def exists(self, objects): return torch.max(objects[: ,0])

    def forall(self, objects): return torch.min(objects[:, 0])
    
    def iota(self, objects): return torch.cat([
        torch.logit(torch.softmax(objects[:, 0], dim = 0).reshape([-1,1]), eps = 1e-6),
          objects[:,1:]
    ], dim = -1)

    def negate(self, logit): return -logit

    def logic_and(self, logit1, logit2): return torch.min(logit1, logit2)

    def logic_or(self, logit1, logit2): return torch.max(logit1, logit2)

    def count(self, objects): return torch.sum(torch.sigmoid(objects[:,0]))

    def color_logits(self, objects):
        red_logits = self.red_mlp(objects[:,1:])
        green_logits = self.green_mlp(objects[:,1:])
        blue_logits = self.blue_mlp(objects[:,1:])
        color_logits = torch.cat([red_logits, green_logits, blue_logits], dim = -1)
        
        #color_logits = torch.logit(stable_softmax(color_logits, dim = 1), eps = 1e-6)

        #print("red logits:",    color_logits[:,0])
        #print("greeen logits:", color_logits[:,1])
        #print("blue logits:",   color_logits[:,2])
        
        return color_logits

    def red(self, objects):        
        red_logits = self.color_logits(objects)[:,0]
        return torch.cat([torch.min(red_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def green(self, objects):
        green_logits = self.color_logits(objects)[:,1]

        return torch.cat([torch.min(green_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)

    def blue(self, objects):
        blue_logits = self.color_logits(objects)[:,2]
        return torch.cat([torch.min(blue_logits, objects[:,0]).reshape([-1,1]),objects[:,1:],], dim = -1)
    
    def shape_logits(self, objects):
        circle_logits = self.circle_mlp(objects[:,1:])
        rectangle_logits = self.rectangle_mlp(objects[:,1:])
        triangle = self.triangle_mlp(objects[:,1:])
        
        shape_logits = torch.cat([circle_logits, rectangle_logits, triangle], dim = 1)
        shape_logits = torch.logit(stable_softmax(shape_logits, dim = 1), eps = 1e-6)
        #print("circle logits: ", shape_logits[:,0])
        #print("rectangle logits: ", shape_logits[:,1])
        #print("triangle logits: ", shape_logits[:,2])
        
        return shape_logits

    def circle(self, objects):
        circle_logits = self.shape_logits(objects)[:, 0]
        return torch.cat([
            torch.min(circle_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = 1)

    def rectangle(self, objects):
        rectangle_logits = self.shape_logits(objects)[:, 1]
        return torch.cat([
            torch.min(rectangle_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = 1)

    def triangle(self, objects):
        triangle_logits = self.shape_logits(objects)[:, 2]
        return torch.cat([
            torch.min(triangle_logits, objects[:,0]).reshape([-1,1]), 
            objects[:,1:]
        ], dim = -1)
    
    def relation_features(self, anchor_object, ref_objects):
        anchor_features = anchor_object[:,1:] # [nxd]
        ref_features = ref_objects[:, 1:] # [nxd]

        anchor_expanded = anchor_features.unsqueeze(1).expand(-1, ref_features.size(0), -1)
        ref_expanded = ref_features.unsqueeze(0).expand(anchor_features.size(0), -1, -1)
        feature_matrix = torch.cat([anchor_expanded, ref_expanded], dim=2)

        return feature_matrix

    def left(self, anchor_object, ref_objects):
        ref_logits = ref_objects[:, 0:1] 
        anchor_logits = anchor_object[:,0:1]

        relation_features = self.relation_features(anchor_object, ref_objects)
        left_logits = self.left_mlp(relation_features)
        #anchor_dist = torch.softmax(anchor_logits, dim=0)
        anchor_dist = torch.sigmoid(anchor_logits)
        #anchor_dist = anchor_logits.unsqueeze(-1).unsqueeze(-1)
        
        n = ref_logits.shape[0]
        #print("anchor logits:", anchor_logits)
        #print("ref logits:", ref_logits)

        #stprint(left_logits)
        #stprint(anchor_dist)
        #output_ref_logits = torch.einsum("nnk,nk -> nk",left_logits, anchor_dist)  # [n, 1]  expectation of ref logits over the anchor object distribution (first dimension)
        output_ref_logits = torch.sum(left_logits * anchor_dist.unsqueeze(0).repeat(n,1,1), dim = 0)
        output_ref_logits = torch.min(output_ref_logits, ref_logits)
        #stprint(output_ref_logits)

        output_ref_objects = torch.cat(
            [output_ref_logits, ref_objects[:, 1:]], dim = -1
        )
        #print("left logits",left_logits.reshape(n,n))
        return output_ref_objects
    
    def right(self, anchor_object, ref_objects):
        ref_logits = ref_objects[:, 0:1] 
        anchor_logits = anchor_object[:,0:1]
        n = ref_logits.shape[0]

        relation_features = self.relation_features(anchor_object, ref_objects)
        right_logits = self.right_mlp(relation_features)
        #anchor_dist = torch.softmax(anchor_logits, dim=0)
        anchor_dist = torch.sigmoid(anchor_logits)
        #anchor_dist = anchor_logits.unsqueeze(-1).unsqueeze(-1)

        #stprint(anchor_dist)
        #stprint( right_logits)
        #output_ref_logits = torch.einsum("nnk,nk -> nk",right_logits, anchor_dist)  # [n, 1]  expectation of ref logits over the anchor object distribution (first dimension)
        output_ref_logits = torch.sum(right_logits * anchor_dist.unsqueeze(0).repeat(n,1,1), dim = 0)
        #stprint(output_ref_logits)
        #stprint(ref_logits)
        output_ref_logits = torch.min(output_ref_logits, ref_logits)
        #stprint(output_ref_logits)

        output_ref_objects = torch.cat(
            [output_ref_logits, ref_objects[:, 1:]], dim = -1
        )
        #print("right logits",right_logits.reshape(n,n))
        return output_ref_objects

objects_executor = ObjectsExecutor(objects_domain)
