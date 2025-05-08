
## Meta Concept Learner
Efficient continual learning of new concept and flexible representation using natural supervision.
![image](outputs/example.png)

## Prerequisites
this section is about the prerequisites

```
conda create -n nscl anaconda
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