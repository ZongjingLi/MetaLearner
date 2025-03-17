import torch
import torch.nn as nn
import math

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128):
        super().__init__()
        self.pre_map = nn.Linear(input_dim, latent_dim)
        self.final_map = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x, adj, residual = True):
        """
        inputs:
            x: [B,N,D], adj: [B,N,N]
        outputs:
            y: [B,N,E]
        """
        x = self.pre_map(x)
        if isinstance(adj, torch.Sparse):
            pass
        else:
            if residual: x = x + torch.bmm(adj, x)
            x = torch.bmm(adj, x)
        x = self.final_map(x)
        return x

class SparseGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super(GraphConvolution).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttention(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim = 128, 
                    with_v = False, retract = True, normalize = True):
        super().__init__()
        self.k_map = nn.Linear(input_dim, latent_dim)
        self.q_map = nn.Linear(input_dim, latent_dim)

        self.v_map = nn.Linear(input_dim, output_dim) if with_v else nn.Identity()


        self.retract = retract
        self.normalize = normalize

    def forward(self, x, adj):
        """
        inputs:x: [B,N,D], adj: [B,N,N]
        outputs:y: [B,N,E]
        """
        if self.normlaize: x = nn.functional(x, p = 2)
        s = math.sqrt(x.shape[-1])
        ks = self.k_map(x) # BxNxDk
        qs = self.q_map(x) # BxNxDq
        attn = torch.bmm(ks/s,qs.transpose(1,2)/s)
        if self.retract: attn -= 0.5
        attn = attn * adj

        vs = self.v_map(x) # BxNxDv
        outputs = torch.bmm(attn, vs)
        if self.normalize: outputs = nn.functional.normalize(outputs, p = 2)
        return outputs