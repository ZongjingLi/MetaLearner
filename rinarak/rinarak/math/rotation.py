import numpy as np
import torch

def rotate(theta, phi):
    return np.cos(theta) * np.sin(phi)

def rotx(t):
    return torch.tensor([
        [1.,     0.,       0.           ],
        [0.,     np.cos(t), -np.sin(t)  ],
        [0.,     np.sin(t), np.cos(t),  ]
    ]).float()

def roty(t):
    return torch.tensor([
        [np.cos(t),      0.,            np.sin(t)],
        [0.,             1.,            0,       ],
        [-np.sin(t),     0,             np.cos(t)],
    ]).float()

def rotz(t):
    return torch.tensor([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t),  0],
        [0,         0,          1],
    ]).float()

def euler_rot(tx,ty,tz):
    return torch.matmul(torch.matmul(rotx(tx),roty(ty)),rotz(tz))

def so2(t, homogenous = False):
    """
    input should be single scalar of angle
    return: 2x2 or 2x3
    """
    if homogenous:
        rot_matrix = np.array([
            [ np.cos(t), np.sin(t), 0],
            [-np.sin(t), np.cos(t), 0],
        ])
    else:
        rot_matrix = np.array([
        [np.cos(t), np.sin(t)],
        [-np.sin(t),np.cos(t)]]
        )
    return rot_matrix

def se2(t,a):
    """
    there is no choice in se2, must use homogenous coords
    return: 2x3
    """
    trans_mat = np.array([
            [ np.cos(t), np.sin(t), a[0]],
            [-np.sin(t), np.cos(t), a[1]],
    ])
    return trans_mat

def pauli_matrices(vec4 = False):
    s1 = np.array([
        [0, 1],
        [1, 0]
    ])
    s2 = np.array([
        [0, -1j],
        [1j, 0]
    ])
    s3 = np.array([
        [1,  0],
        [0, -1]
    ])
    if vec4:
        s0 = np.ones([2,2])
        return s0,s1,s2,s3
    return s1,s2,s3