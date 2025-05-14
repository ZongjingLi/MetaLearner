
## Meta Concept Learner
Efficient continual learning of new concept and flexible representation using natural supervision.
![image](outputs/example.png)

## Prerequisites
this section is about the prerequisites

```
conda create -n prototype anaconda
conda install pytorch torchvision -c pytorch
```

## Dataset preparation
this section is about the dataset generation.
we can find the scripts to generate the data used for our experiment under the directory of 
`scripts/data_gen/...`


## Training and evaluation
this section is about training and testing of the model.

## Commands for the Model
All the model checkpoints are saved under the directory `outputs/checkpoints`. A model checkpoint with $\texttt{name}$ will have a folder of same name under the previous directory. This stores the the domain learneable parameters under the `domains` directory and reductions under the `frames` directory.
```
Model
├── config.yaml
├── core_vocab.txt
├── lexicon_entries.pth
├── lexicon_weights.pth
|
├── domains
│   ├── Color.pth
│   ├── Order.pth
│   ├── Scene.pth
|   └── Integer.pth
|
└── frames
    ├── images
    ├── questions.json
    ├── scenes-raw.json
    └── vocab.json
```
### Dataset and environment config
dataset and  environment configs are saved under the `configs/dataset_config.yaml` and `configs/env_config.yaml`. In the dataset config it stored the name of the dataset and path of to load this dataset and the corresponding $\texttt{getter()}$ method

### Create blank MetaLearner
`scripts/grounding/create_prototype.sh` this will create the prototype MetaLearner using the vocab and domain config under the `data/core_vocab.txt` and `configs/core_domains.yaml`. This command will load the vocabulary and domains specified in the  $\texttt{core-domains.yaml}$  then 

### Learn vocabulary-domain association
scripts under the dir `scripts/` will load the specified model

Fun Fun Fun
https://sites.google.com/view/virtualtoolsgame

## Framework Details

### Parser and CCG
In this work we use the categorical-combinatoric-grammar to perform the parsing the model [G2L2](https://proceedings.neurips.cc/paper_files/paper/2021/file/4158f6d19559955bae372bb00f6204e4-Paper.pdf).


## Experiment Results
| Method | Acc | Center | 2x2Grid | 3x3Grid | L-R | U-D | O-IC | O-IG |
|--------|-----|--------|---------|---------|-----|-----|------|------|
| LSTM | 13.07% | 13.19% | 14.13% | 13.69% | 12.84% | 12.35% | 12.15% | 12.99% |
| WReN | 14.69% | 13.09% | 28.62% | 28.27% | 7.49% | 6.34% | 8.38% | 10.56% |
| CNN | 36.97% | 33.58% | 30.30% | 33.53% | 39.43% | 41.26% | 43.20% | 37.54% |
| ResNet | 53.43% | 52.82% | 41.86% | 44.29% | 58.77% | 60.16% | 63.19% | 53.12% |
| LSTM+DRT | 13.96% | 14.29% | 15.08% | 14.09% | 13.79% | 13.24% | 13.99% | 13.29% |
| WReN+DRT | 15.02% | 15.38% | 23.26% | 29.51% | 6.99% | 8.43% | 8.93% | 12.35% |
| CNN+DRT | 39.42% | 37.30% | 30.06% | 34.57% | 45.49% | 45.54% | 45.93% | 37.54% |
| ResNet+DRT | 59.56% | 58.08% | 46.53% | 50.40% | 65.82% | 67.11% | 69.09% | 60.11% |
| Human | 84.41% | 95.45% | 81.82% | 79.55% | 86.36% | 81.81% | 86.36% | 81.81% |
| Solver | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |

### Related Works
**Neuro-Symbolic Concept Learner**

[NS-CL](https://arxiv.org/pdf/1904.12584) The Neuro-Symbolic Concept Learner: Interpreting Scenes, Words and Sentences from Natural Supervision : the inital work of neuro-symbolic concept learner, joint learning with natural supervision 

[G2L2](https://proceedings.neurips.cc/paper_files/paper/2021/file/4158f6d19559955bae372bb00f6204e4-Paper.pdf) : associate word with CCG lexicon entries with natural supervision 

[FALCON](https://arxiv.org/pdf/2203.16639) : fast visual concept learning by integrating images, linguistic descriptions and conceptual relations. It provides a meta-learning framework that can fast learning new visual concepts with just one or few examples guided by mulitple naturally occuring data streams.

[Mechanisms](https://arxiv.org/pdf/2311.03293): Learining Reusable Planning Stratagies. A specific strategy is specified by the contact mode. Parameters of each stratagey is grounded by self-replay.

[DCL](https://arxiv.org/pdf/2103.16564) Grounded Physical Concepts of Objects and Events Through Dynamic Visual Reasoning: learning to ground visual concepts and dynamic concepts. This framework provides a useful DSL for the learining of dyanmic concepts.