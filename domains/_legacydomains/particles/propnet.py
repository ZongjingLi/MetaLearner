'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-07-29 03:44:46
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-07-29 03:44:49
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data  import Data,Batch
from torch_geometric.nn    import max_pool_x, GraphConv
from torch_scatter import scatter_mean,scatter_max, scatter_sum

class ParticleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class RelationEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class EffectEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        return self.model(x)

class AggregateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        x = self.model(x)
        return x

class ParticleDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.output_size = output_dim

        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.CELU()
    def forward(self, x):
        x = self.linear_1(self.relu(self.linear_0(x)))
        #x = self.linear_1(self.linear_0(x))
        return x
    
def build_edges(positions, theshold = 1.0, offset = 0):
    row, col = [], []
    for i,pos1 in enumerate(positions):
        for j,pos2 in enumerate(positions):
            dist = torch.norm(pos1 - pos2)
            if i!=j:#dist < thershold and i != j:
                row.append(i + offset)
                col.append(j + offset)
    return [row, col]

def tensor_product(t1, t2, dim = -1):
    n, d1 = t1.shape
    m, d2 = t2.shape
    out = torch.cat([
        t1.unsqueeze(1).repeat(1,m,1), t2.unsqueeze(0).repeat(n,1,1)
    ], dim = dim)
    return out

class ParticlePropagator(nn.Module):
    def __init__(self, pos_dim, attr_dim, relation_dim, state_dim, roll_num = 10):
        super().__init__()
        self.pos_dim = pos_dim
        self.attr_dim = attr_dim
        self.state_dim = state_dim
        hidden_dim = 300
        input_dim = pos_dim + attr_dim
        """encoder of particles and effect propagations"""
        self.particle_encoder = ParticleEncoder(input_dim, hidden_dim, state_dim)
        self.relation_encoder = RelationEncoder(pos_dim + pos_dim , hidden_dim, state_dim)
        self.effect_encoder = EffectEncoder(state_dim  * 3, hidden_dim, state_dim)

        self.aggregate_encoder = AggregateEncoder(state_dim * 3, hidden_dim, state_dim)
        self.graph_conv = GraphConv(state_dim, state_dim)

        """decoder of particle states and change of (attriubtes not changed)"""
        self.particle_decoder = ParticleDecoder(state_dim * 2, hidden_dim, pos_dim + attr_dim)


    def forward(self, batch_inputs, edges = None, hidden_states = None, steps = 2, device = "cpu"):
        """"""
        batch_partition = [inputs.shape[0] for inputs in batch_inputs]
        prefix = [sum(batch_partition[:i+1]) for i in range(len(batch_partition))]
        prefix.insert(0, 0)
        n = sum(batch_partition)
        if hidden_states is None: hidden_states = torch.zeros([n, self.state_dim])
        if edges is None: edges = [build_edges(inputs, offset = 0) for b,inputs in enumerate(batch_inputs)]


        graph_in = Batch.from_data_list([Data(x,torch.tensor(edges[i]).long())
                                                for i,x in enumerate(batch_inputs)])


        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch
        #x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)
        row, col = edge_index
    
        """1. initalize the propagation state as the 0 state"""
        flatten_inputs = torch.cat([inputs for inputs in batch_inputs], dim = 0)
        flatten_coords = flatten_inputs[:,:2]
        matrix_coords = tensor_product(flatten_coords, flatten_coords)
        encoded_particle_states = self.particle_encoder(flatten_inputs) #(n,d)
        encoded_relation_states = self.relation_encoder(matrix_coords) # (n, n, d)

        """maintain the hidden states"""
        hidden_states_history = [hidden_states]
        for l in range(steps):
            row, col = edge_index # (m,) (m,)
            effect_states = self.effect_encoder(
                torch.cat([
                        tensor_product(hidden_states_history[-1], hidden_states_history[-1]),\
                        encoded_relation_states], dim = -1)
                ) #(m,d)

            sum_effect = torch.sum(effect_states, dim = 0)
            #print(sum_effect.shape, encoded_particle_states.shape, hidden_states_history[-1].shape)
            hidden_states_history.append(self.aggregate_encoder(torch.cat([
                encoded_particle_states,sum_effect, hidden_states_history[-1]], dim = -1) ) )

        predict_pos = self.particle_decoder(
            torch.cat([ encoded_particle_states, hidden_states_history[-1] ], dim = -1)
            ) + flatten_inputs
        partitioned_pos = [predict_pos[prefix[i] : prefix[i+1]] for i in range(len(prefix) - 1)]
        return predict_pos, partitioned_pos, edges

    def predict(self, batch_inputs, edges = None, rolls = 10):
        outputs = []
        for step in range(rolls):
            predictions, batch_inputs, edges = self(batch_inputs)
            outputs.append(batch_inputs)
        return outputs



def train(model, dataset, epochs = 2000):
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2)
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
    min_loss = 1000
    for epoch in range(epochs):
        for idx in range(len(dataset)):
            sample = dataset[idx]
            batch_gt = sample["states"]
            N, T, D = batch_gt[0].shape
            loss = 0.0
            inputs = [gt[:,0,:] for gt in batch_gt]
            outputs = model.predict(inputs, rolls = T)
            for t in range(T):
                for b in range(len(batch_gt)):
                    loss += nn.functional.mse_loss(outputs[t][b], batch_gt[b][:,t,:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                torch.save(model.state_dict(), "propnet_min.pth")
                min_loss = loss
        print("epoch:%d loss: %.2f"%(epoch+1, loss))
    print(min_loss)
    return model

import random

class AcherusDataset(nn.Module):
    def __init__(self, root_dir = "/Users/melkor/Documents/datasets", split = "train", length = 20):
        super().__init__()
        self.dir = root_dir + f"/Acherus/{split}/"
        self.length = length
    
    def __len__(self): return 1

    def __getitem__(self, idx):
        sample = {}
        states = np.load(self.dir + f"{idx}.npy")
        sample_length = states.shape[1]
        sample["states"] = []
        for i in range(5):
            start = np.random.randint(0, sample_length - self.length ) * 1
            sample["states"].append(torch.tensor(states[:, start: start + self.length, :]).clamp(-1.0,1.0))
        return sample

if __name__ == "__main__":
    
    #positions = [torch.tensor([[0.3, 0.2+.2**2], [0.3, 0.8 -.3**2]]), torch.tensor([[0.3, 0.2], [0.8, 0.2], [0.2, 0.3]]) ]
    """
    t_steps = 10
    x1 = torch.linspace(0.1, .8, t_steps)[..., None]
    y1 = .2 - x1 * x1 * 0.2
    d1 = torch.cat([x1, y1], dim = -1)[None, ...]

    x2 = torch.linspace(0.1, .8, t_steps)[..., None]
    y2 = torch.sin(x2 * 1)
    d2 = torch.cat([x2, y2], dim = -1)[None, ...]

    parabolic_data = torch.cat([d1, d2], dim = 0)

    #train(propnet, [parabolic_data])
    #torch.save(propnet.state_dict(), "propnet_demo.pth")
    propnet.load_state_dict(torch.load("domains/particles/propnet_demo.pth"))

    positions = [parabolic_data[:,0,:]]
    """

    
    import time
    import taichi as ti

    dataset =  AcherusDataset(length = 40)
    propnet = ParticlePropagator(2, 6, 1, 200)
    #propnet.load_state_dict(torch.load("propnet_min.pth"))
    #train(propnet, dataset, epochs = 5000)
    propnet.load_state_dict(torch.load("propnet_min.pth"))
    #torch.save(propnet.state_dict(), "propnet_demo.pth")
    


    gui = ti.GUI('Particles', res=(400, 400))
    def draw_rectangle(gui, center, side = .05):
        gui.rect(topleft=[center[0] - side, center[1] - side], bottomright= [center[0] + side, center[1] + side], radius = 2, color=0xCC0000)
    gui.background_color = 0xFFFFFF
    gui.show()
    itrs = 0

    positions = [
        torch.tensor([0.4, 0.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])[None,...],
    ]
    positions = [dataset[0]["states"][0][:,0,:]]


    while itrs < 1100:
        predictions, next_positions, edges = propnet(positions)
        #gui.circle(pos=center, radius=30, color=0xED553B)
        for center in positions[0]:
            draw_rectangle(gui,  center)
        
        row, col = edges[0]
        gui.lines(begin=positions[0][row,:2].detach().numpy(), 
                  end=positions[0][col,:2].detach().numpy(), radius=2, color=0x068587)
        gui.circles(dataset[0]["states"][0][0,:,:2].detach().numpy(), radius=2, color=0xED553B)
        gui.circles(dataset[0]["states"][0][1,:,:2].detach().numpy(), radius=2, color=0xED553B)

        #time.sleep(.1)
    
        gui.show(f"outputs/frames/{itrs}.png")

        #if len(row) < 1: time.sleep(.1)
        #print(positions)
        positions = next_positions
        itrs += 1