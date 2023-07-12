import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_concepts(concepts, model):
    entailment = model.entailment
    registry = model.box_registry
    executor = model.executor
    DiG = nx.DiGraph()
    for c in concepts: DiG.add_node(c)
    for prior_concept in concepts:
        prior_emb = executor.get_concept_embedding(prior_concept)
        for posterior_concept in concepts:
            post_emb = executor.get_concept_embedding(posterior_concept)
            
            entail_prob = torch.sigmoid(entailment(post_emb, prior_emb))

            DiG.add_edge(prior_concept, posterior_concept, weight = entail_prob.detach().squeeze().numpy())


    fig = plt.figure(figsize=plt.figaspect(1/1))
    ax = fig.add_subplot(111, projection="3d"); rang = 1.0

    ax.set_zlim(-rang,rang);ax.set_xlim(-rang,rang);ax.set_ylim(-rang,rang)
        # make the panes transparent
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()


    pos = nx.spring_layout(DiG, dim=3, seed=779)  
    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(DiG)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in DiG.edges()])
    weights = np.array([weight for u, v, weight in DiG.edges(data="weight")])

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w")

    for i in range(len(concepts)):
        x,y,z = node_xyz[i]
        ax.text(x,y,z,concepts[i])

    # Plot the edges
    for i in range(len(edge_xyz)):
        vizedge = edge_xyz[i]
        ax.plot(*vizedge.T, color="tab:gray", alpha = weights[i])

"""
    pos = nx.spring_layout(DiG, seed=7)
    esmall = [(u, v) for (u, v, d) in DiG.edges(data=True) if d["weight"] <= 1.5]
    nx.draw_networkx_edges(
    DiG, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

"""