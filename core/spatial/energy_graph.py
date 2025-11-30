# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2024-10-15 07:17:22
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-03 03:49:43
import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat
from typing import Dict, Tuple, List, Union

def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b

## Basic functions used by all models

class ModelMixin:
    def rand_input(self, batchsize, input_dims = None):
        if input_dims is None:
            assert hasattr(self, 'input_dims'), 'Model must have "input_dims" attribute!'
            return torch.randn((batchsize,) + self.input_dims)
        return torch.randn((batchsize,) + input_dims)

    # Currently predicts eps, override following methods to predict, for example, x0
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):

        return loss()(eps, self(x0 + sigma * eps, sigma, cond=cond)["gradient"])

    def predict_eps(self, x, sigma, cond=None):
        return self(x, sigma, cond=cond)["gradient"]

    def predict_eps_cfg(self, x, sigma, cond, cfg_scale):
        b = x.shape[0]
        if sigma.flatten().shape[0] == 1:
            sigma = sigma.reshape([1,1]).repeat(b,1)
        if cond is None:
            return self.predict_eps(x, sigma)
        else:
            return self.predict_eps(x, sigma, cond)
        assert sigma.shape == tuple(), 'CFG sampling only supports singleton sigma!'
        uncond = torch.full_like(cond, self.cond_embed.null_cond) # (B,)
        eps_cond, eps_uncond = self.predict_eps(                  # (B,), (B,)
            torch.cat([x, x]), sigma, torch.cat([cond, uncond])   # (2B,)
        ).chunk(2)
        return eps_cond + cfg_scale * (eps_cond - eps_uncond)

def sigma_log_scale(batches, sigma, scaling_factor):
    if sigma.shape == torch.Size([]):
        sigma = sigma.unsqueeze(0).repeat(batches)
    else:
        assert sigma.shape == (batches,), 'sigma.shape == [] or [batches]!'
    return torch.log(sigma)*scaling_factor

def get_sigma_embeds(batches, sigma, scaling_factor=0.5):
    s = sigma_log_scale(batches, sigma, scaling_factor).unsqueeze(1)
    return torch.cat([torch.sin(s), torch.cos(s)], dim=1)

# A simple embedding that works just as well as usual sinusoidal embedding
class SigmaEmbedderSinCos(nn.Module):
    def __init__(self, hidden_size, scaling_factor=0.5):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, batches, sigma):
        sig_embed = get_sigma_embeds(batches, sigma, self.scaling_factor) # (B, 2)
        return self.mlp(sig_embed)                                        # (B, D)


## Modifiers for models, such as including scaling or changing model predictions

def alpha(sigma):
    return 1/(1+sigma**2)

# Scale model input so that its norm stays constant for all sigma
def Scaled(cls: ModelMixin):
    def forward(self, x, sigma, cond=None):
        return cls.forward(self, x * alpha(sigma).sqrt(), sigma, cond=cond)
    return type(cls.__name__ + 'Scaled', (cls,), dict(forward=forward))

# Train model to predict x0 instead of eps
def PredX0(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        return loss()(x0, self(x0 + sigma * eps, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        x0_hat = self(x, sigma, cond=cond)
        return (x - x0_hat)/sigma
    return type(cls.__name__ + 'PredX0', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))

# Train model to predict v (https://arxiv.org/pdf/2202.00512.pdf) instead of eps
def PredV(cls: ModelMixin):
    def get_loss(self, x0, sigma, eps, cond=None, loss=nn.MSELoss):
        xt = x0 + sigma * eps
        v = alpha(sigma).sqrt() * eps - (1-alpha(sigma)).sqrt() * x0
        return loss()(v, self(xt, sigma, cond=cond))
    def predict_eps(self, x, sigma, cond=None):
        v_hat = self(x, sigma, cond=cond)
        return alpha(sigma).sqrt() * (v_hat + (1-alpha(sigma)).sqrt() * x)
    return type(cls.__name__ + 'PredV', (cls,),
                dict(get_loss=get_loss, predict_eps=predict_eps))

## Common functions for other models

class CondSequential(nn.Sequential):
    def forward(self, x, cond):
        for module in self._modules.values():
            x = module(x, cond)
        return x

class Attention(nn.Module):
    def __init__(self, head_dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # (B, N, D) -> (B, N, D)
        # N = H * W / patch_size**2, D = num_heads * head_dim
        q, k, v = rearrange(self.qkv(x), 'b n (qkv h k) -> qkv b h n k',
                            h=self.num_heads, k=self.head_dim)
        x = rearrange(F.scaled_dot_product_attention(q, k, v),
                      'b h n k -> b n (h k)')
        return self.proj(x)

# Embedding table for conditioning on labels assumed to be in [0, num_classes),
# unconditional label encoded as: num_classes
class CondEmbedderLabel(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout_prob=0.1):
        super().__init__()
        self.embeddings = nn.Embedding(num_classes + 1, hidden_size)
        self.null_cond = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels): # (B,) -> (B, D)
        if self.training:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.null_cond, labels)
        return self.embeddings(labels)

## Simple MLP for toy examples

class TimeInputMLP(nn.Module, ModelMixin):
    def __init__(self, dim=2, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + 2,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], dim))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma, cond=None):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2
        nn_input = torch.cat([x, sigma_embeds], dim=1)               # shape: b x (dim + 2)
        return self.net(nn_input)

class TimeInputEnergyMLP(nn.Module, ModelMixin):
    def __init__(self, dim=2, attr_dim = 0, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        layers = []
        for in_dim, out_dim in pairwise((dim + 2,) + hidden_dims):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.net = nn.Sequential(*layers)
        self.input_dims = (dim,)

    def forward(self, x, sigma, cond=None, energy_only = True):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        if not energy_only:
            x.requires_grad = True
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x 2

        nn_input = torch.cat([x, sigma_embeds], dim=1).to(next(self.parameters()).device)              # shape: b x (dim + 2)

        energy = self.net(nn_input)
        
        if not energy_only:
            grad = torch.autograd.grad(energy.flatten().sum(), x,
	        retain_graph = True, create_graph=True)[0]
        else: grad = 0.0

        return {"energy":energy, "gradient":grad}



class PointEnergyMLP(nn.Module, ModelMixin):
    def __init__(self, constraints : Dict[str, Union[List[Union[int, Tuple]]]], dim=2, hidden_dims=(16,128,256,128,16)):
        super().__init__()
        self.energies = nn.ModuleDict({})
        for name in constraints:
            arity = len(constraints[name])
            state_attr_config = constraints[name]
            input_dim = 0
            attr_dim  = 0
            for state_attr in state_attr_config:
                if isinstance(state_attr, Tuple): input_dim += state_attr[0] + state_attr[1]; attr_dim += state_attr[1]
                elif isinstance(state_attr, int): input_dim += state_attr
                else: raise ValueError(state_attr)
            self.energies[name] = TimeInputEnergyMLP(input_dim, attr_dim = attr_dim)
            input_dim = int(input_dim / arity)
        self.input_dims = (input_dim, )

    def forward(self, x, sigma, attr = None, cond = None):

        assert cond is not None, """cond requires edge parameters"""
        total_energy = 0.0
        x.requires_grad = True
        
        """handle the case of inputs are conditioned by the attributes"""
        if attr is not None:
            attr.required_grad = False
            attr_cond_inputs = torch.cat([x, attr], dim = -1)
        else:
            attr_cond_inputs = x

        for edge in cond["edges"]:
            obj_idx = edge[:-1]
            type_name = edge[-1]

            x_inputs = attr_cond_inputs[:, obj_idx, :]#x[:,obj_idx, :]

            b, n, d= x_inputs.shape

            x_inputs = x_inputs.reshape([1, n * d])

            sigma_inputs = sigma


            comp = self.energies[type_name](x_inputs, sigma_inputs)

            total_energy = total_energy + comp["energy"]
        total_energy /= len(cond["edges"])

        grad = torch.autograd.grad(total_energy.flatten().sum(), x,retain_graph = True, create_graph=True)[0]
        return {"energy" : total_energy, "gradient" : grad}


## Ideal denoiser defined by a dataset

def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

class IdealDenoiser(nn.Module, ModelMixin):
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.data = torch.stack([dataset[i] for i in range(len(dataset))])
        self.input_dims = self.data.shape[1:]

    def forward(self, x, sigma, cond=None):
        data = self.data.to(x)                                                         # shape: db x c1 x ... x cn
        x_flat = x.flatten(start_dim=1)
        d_flat = data.flatten(start_dim=1)
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        assert sigma.shape == tuple() or sigma.shape[0] == xb, \
            f'sigma must be singleton or have same batch dimension as x! {sigma.shape}'
        # sq_diffs: ||x - x0||^2
        sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T # shape: db x xb
        weights = torch.nn.functional.softmax(-sq_diffs/2/sigma.squeeze()**2, dim=0)             # shape: db x xb
        eps = torch.einsum('ij,i...->j...', weights, data)                             # shape: xb x c1 x ... x cn
        return (x - eps) / sigma
