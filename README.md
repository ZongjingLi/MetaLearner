
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




\tikzset{every picture/.style={line width=0.75pt}} %set default line width to 0.75pt        

\begin{tikzpicture}[x=0.75pt,y=0.75pt,yscale=-1,xscale=1]
%uncomment if require: \path (0,2406); %set diagram left start at 0, and has height of 2406

%Shape: Circle [id:dp9765463238450909] 
\draw   (65.33,43.17) .. controls (65.33,35.16) and (71.83,28.67) .. (79.83,28.67) .. controls (87.84,28.67) and (94.33,35.16) .. (94.33,43.17) .. controls (94.33,51.17) and (87.84,57.67) .. (79.83,57.67) .. controls (71.83,57.67) and (65.33,51.17) .. (65.33,43.17) -- cycle ;
%Shape: Triangle [id:dp6907013024799116] 
\draw   (124,197) -- (159,237) -- (89,237) -- cycle ;
%Shape: Triangle [id:dp7033729206866222] 
\draw   (44.67,197) -- (79.67,237) -- (9.67,237) -- cycle ;
%Straight Lines [id:da10095267380980544] 
\draw    (44.67,197) -- (80,175) ;
%Straight Lines [id:da6061799742424681] 
\draw    (124,197) -- (80,175) ;
%Shape: Rectangle [id:dp8908090971158418] 
\draw   (71,129.33) -- (88.67,129.33) -- (88.67,147.67) -- (71,147.67) -- cycle ;
%Shape: Rectangle [id:dp7439559188339517] 
\draw   (71,79.33) -- (88.67,79.33) -- (88.67,97.67) -- (71,97.67) -- cycle ;
%Straight Lines [id:da6251460614792923] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ]   (80.67,128.33) -- (80.67,99) ;
\draw [shift={(80.67,113.67)}, rotate = 90] [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=0.75]    (0,5.59) -- (0,-5.59)   ;
%Straight Lines [id:da8528990531138099] 
\draw    (80,175) -- (80.67,148.33) ;
\draw [shift={(80.46,156.67)}, rotate = 91.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da7298112626235547] 
\draw    (80.67,78.83) -- (80.67,58) ;
\draw [shift={(80.67,63.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp6959267415212689] 
\draw   (183.67,43.83) .. controls (183.67,35.83) and (190.16,29.33) .. (198.17,29.33) .. controls (206.17,29.33) and (212.67,35.83) .. (212.67,43.83) .. controls (212.67,51.84) and (206.17,58.33) .. (198.17,58.33) .. controls (190.16,58.33) and (183.67,51.84) .. (183.67,43.83) -- cycle ;
%Shape: Rectangle [id:dp6399116692978233] 
\draw   (189.33,130) -- (207,130) -- (207,148.33) -- (189.33,148.33) -- cycle ;
%Shape: Rectangle [id:dp3870528279456751] 
\draw   (189.33,80) -- (207,80) -- (207,98.33) -- (189.33,98.33) -- cycle ;
%Straight Lines [id:da6226406542730587] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (199,129) -- (199,99.67) ;
\draw [shift={(199,109.33)}, rotate = 90] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da4333955245198162] 
\draw    (198.17,79.17) -- (198.17,58.33) ;
\draw [shift={(198.17,63.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Rounded Rect [id:dp15425465560655338] 
\draw   (249.67,69.33) .. controls (249.67,62.71) and (255.04,57.33) .. (261.67,57.33) -- (297.67,57.33) .. controls (304.29,57.33) and (309.67,62.71) .. (309.67,69.33) -- (309.67,108.33) .. controls (309.67,114.96) and (304.29,120.33) .. (297.67,120.33) -- (261.67,120.33) .. controls (255.04,120.33) and (249.67,114.96) .. (249.67,108.33) -- cycle ;
%Shape: Circle [id:dp9077450929890376] 
\draw   (258.33,79) .. controls (258.33,72.74) and (263.41,67.67) .. (269.67,67.67) .. controls (275.93,67.67) and (281,72.74) .. (281,79) .. controls (281,85.26) and (275.93,90.33) .. (269.67,90.33) .. controls (263.41,90.33) and (258.33,85.26) .. (258.33,79) -- cycle ;
%Straight Lines [id:da000912666784433247] 
\draw    (207.67,138.33) -- (259,112.33) ;
\draw [shift={(237.79,123.07)}, rotate = 153.14] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da38959974282822163] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (253.67,72.33) -- (212.33,51) ;
\draw [shift={(228.56,59.37)}, rotate = 27.3] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp14087881692622917] 
\draw   (278.33,99) .. controls (278.33,92.74) and (283.41,87.67) .. (289.67,87.67) .. controls (295.93,87.67) and (301,92.74) .. (301,99) .. controls (301,105.26) and (295.93,110.33) .. (289.67,110.33) .. controls (283.41,110.33) and (278.33,105.26) .. (278.33,99) -- cycle ;
%Shape: Circle [id:dp1466922730716873] 
\draw   (195,185.5) .. controls (195,177.49) and (201.49,171) .. (209.5,171) .. controls (217.51,171) and (224,177.49) .. (224,185.5) .. controls (224,193.51) and (217.51,200) .. (209.5,200) .. controls (201.49,200) and (195,193.51) .. (195,185.5) -- cycle ;
%Shape: Rectangle [id:dp36975836039499343] 
\draw   (274,222.33) -- (291.67,222.33) -- (291.67,240.67) -- (274,240.67) -- cycle ;
%Shape: Rectangle [id:dp5848115172107882] 
\draw   (202.67,221.67) -- (220.33,221.67) -- (220.33,240) -- (202.67,240) -- cycle ;
%Straight Lines [id:da18415519345656883] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (271.33,231.67) -- (220.67,231.67) ;
\draw [shift={(241,231.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da934289852970762] 
\draw    (211.5,220.83) -- (211.5,200) ;
\draw [shift={(211.5,205.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da8398327329852051] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (266,184.67) -- (226.33,184.67) ;
\draw [shift={(241.17,184.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp7924333540767734] 
\draw   (267,185.5) .. controls (267,177.49) and (273.49,171) .. (281.5,171) .. controls (289.51,171) and (296,177.49) .. (296,185.5) .. controls (296,193.51) and (289.51,200) .. (281.5,200) .. controls (273.49,200) and (267,193.51) .. (267,185.5) -- cycle ;
%Straight Lines [id:da14827500721438147] 
\draw    (281.33,222.33) -- (281.5,200) ;
\draw [shift={(281.45,206.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp5401402881008162] 
\draw   (371,69.5) .. controls (371,61.49) and (377.49,55) .. (385.5,55) .. controls (393.51,55) and (400,61.49) .. (400,69.5) .. controls (400,77.51) and (393.51,84) .. (385.5,84) .. controls (377.49,84) and (371,77.51) .. (371,69.5) -- cycle ;
%Shape: Rectangle [id:dp33010260315031115] 
\draw   (450,106.33) -- (467.67,106.33) -- (467.67,124.67) -- (450,124.67) -- cycle ;
%Shape: Rectangle [id:dp7598210704835939] 
\draw   (378.67,105.67) -- (396.33,105.67) -- (396.33,124) -- (378.67,124) -- cycle ;
%Straight Lines [id:da23362596975604055] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (447.33,115.67) -- (396.67,115.67) ;
\draw [shift={(417,115.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da29809137667053154] 
\draw    (387.5,104.83) -- (387.5,84) ;
\draw [shift={(387.5,89.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da5939627809213179] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (439.67,69.5) -- (400,69.5) ;
\draw [shift={(414.83,69.5)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp6239382382341754] 
\draw   (443,69.5) .. controls (443,61.49) and (449.49,55) .. (457.5,55) .. controls (465.51,55) and (472,61.49) .. (472,69.5) .. controls (472,77.51) and (465.51,84) .. (457.5,84) .. controls (449.49,84) and (443,77.51) .. (443,69.5) -- cycle ;
%Straight Lines [id:da3815829037211136] 
\draw    (457.33,106.33) -- (457.5,84) ;
\draw [shift={(457.45,90.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp6417309172163184] 
\draw   (522,69.33) -- (539.67,69.33) -- (539.67,87.67) -- (522,87.67) -- cycle ;
%Shape: Circle [id:dp3544627340048807] 
\draw   (515,32.5) .. controls (515,24.49) and (521.49,18) .. (529.5,18) .. controls (537.51,18) and (544,24.49) .. (544,32.5) .. controls (544,40.51) and (537.51,47) .. (529.5,47) .. controls (521.49,47) and (515,40.51) .. (515,32.5) -- cycle ;
%Straight Lines [id:da5152510986980194] 
\draw    (529.33,69.33) -- (529.5,47) ;
\draw [shift={(529.45,53.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp2815972566669247] 
\draw   (530,173.33) -- (547.67,173.33) -- (547.67,191.67) -- (530,191.67) -- cycle ;
%Shape: Circle [id:dp5150095734242284] 
\draw   (523,136.5) .. controls (523,128.49) and (529.49,122) .. (537.5,122) .. controls (545.51,122) and (552,128.49) .. (552,136.5) .. controls (552,144.51) and (545.51,151) .. (537.5,151) .. controls (529.49,151) and (523,144.51) .. (523,136.5) -- cycle ;
%Straight Lines [id:da05521020567411883] 
\draw    (537.33,173.33) -- (537.5,151) ;
\draw [shift={(537.45,157.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp6250424754035484] 
\draw   (610,99.33) -- (627.67,99.33) -- (627.67,117.67) -- (610,117.67) -- cycle ;
%Shape: Circle [id:dp3957167715853456] 
\draw   (603,62.5) .. controls (603,54.49) and (609.49,48) .. (617.5,48) .. controls (625.51,48) and (632,54.49) .. (632,62.5) .. controls (632,70.51) and (625.51,77) .. (617.5,77) .. controls (609.49,77) and (603,70.51) .. (603,62.5) -- cycle ;
%Straight Lines [id:da3656702753279064] 
\draw    (617.33,99.33) -- (617.5,77) ;
\draw [shift={(617.45,83.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp47108354953008247] 
\draw   (410,204.33) -- (427.67,204.33) -- (427.67,222.67) -- (410,222.67) -- cycle ;
%Shape: Circle [id:dp40111762551648433] 
\draw   (403,167.5) .. controls (403,159.49) and (409.49,153) .. (417.5,153) .. controls (425.51,153) and (432,159.49) .. (432,167.5) .. controls (432,175.51) and (425.51,182) .. (417.5,182) .. controls (409.49,182) and (403,175.51) .. (403,167.5) -- cycle ;
%Straight Lines [id:da3122046374990137] 
\draw    (417.33,204.33) -- (417.5,182) ;
\draw [shift={(417.45,188.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da20505801714565774] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (515,38.5) -- (472,69.5) ;
\draw [shift={(489.44,56.92)}, rotate = 324.21] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da25630631081918276] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (603,62.5) -- (544,32.5) ;
\draw [shift={(569.04,45.23)}, rotate = 26.95] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da11724957970764449] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (603,64.5) -- (552,133) ;
\draw [shift={(574.51,102.76)}, rotate = 306.67] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6116106697822685] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (446,84) -- (422,155.5) ;
\draw [shift={(432.41,124.49)}, rotate = 288.56] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da5412759302325052] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (457,129) -- (429,210) ;
\draw [shift={(441.37,174.23)}, rotate = 289.07] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da2260953797322831] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (529,183) -- (468,116) ;
\draw [shift={(495.13,145.8)}, rotate = 47.68] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da5739657471037785] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (609,108) -- (549,181) ;
\draw [shift={(575.83,148.36)}, rotate = 309.42] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da4463201068222189] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (520,130) -- (471,78) ;
\draw [shift={(492.07,100.36)}, rotate = 46.7] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da3680071251161705] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (521,79) -- (468,115) ;
\draw [shift={(490.36,99.81)}, rotate = 325.81] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da7767851799089178] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (609,108) -- (543.5,77.75) ;
\draw [shift={(571.71,90.78)}, rotate = 24.79] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da05402473962772936] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=1.5]    (490.67,1828.67) -- (602,1827.67) ;
\draw [shift={(553.13,1828.11)}, rotate = 179.49] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (11.61,-5.58) -- (0,0) -- (11.61,5.58) -- cycle    ;
%Straight Lines [id:da4819247967937257] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=1.5]  [dash pattern={on 5.63pt off 4.5pt}]  (489.33,1735.58) -- (599.5,1736.25) ;
\draw [shift={(551.22,1735.96)}, rotate = 180.35] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (11.61,-5.58) -- (0,0) -- (11.61,5.58) -- cycle    ;
%Shape: Circle [id:dp9718139834058843] 
\draw   (457.83,1735.58) .. controls (457.83,1726.88) and (464.88,1719.83) .. (473.58,1719.83) .. controls (482.28,1719.83) and (489.33,1726.88) .. (489.33,1735.58) .. controls (489.33,1744.28) and (482.28,1751.33) .. (473.58,1751.33) .. controls (464.88,1751.33) and (457.83,1744.28) .. (457.83,1735.58) -- cycle ;
%Shape: Circle [id:dp24977823924308473] 
\draw   (599.5,1736.25) .. controls (599.5,1727.55) and (606.55,1720.5) .. (615.25,1720.5) .. controls (623.95,1720.5) and (631,1727.55) .. (631,1736.25) .. controls (631,1744.95) and (623.95,1752) .. (615.25,1752) .. controls (606.55,1752) and (599.5,1744.95) .. (599.5,1736.25) -- cycle ;
%Straight Lines [id:da09190128033343337] 
\draw    (474.33,1813.33) -- (473.58,1751.33) ;
\draw [shift={(473.9,1777.33)}, rotate = 89.31] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da030316255851138285] 
\draw [line width=0.75]    (485.67,1816.67) .. controls (486.24,1814.38) and (487.66,1813.52) .. (489.95,1814.09) .. controls (492.24,1814.66) and (493.66,1813.8) .. (494.23,1811.51) .. controls (494.8,1809.22) and (496.23,1808.36) .. (498.52,1808.93) .. controls (500.81,1809.5) and (502.23,1808.64) .. (502.8,1806.35) .. controls (503.37,1804.06) and (504.79,1803.2) .. (507.08,1803.77) .. controls (509.37,1804.34) and (510.8,1803.48) .. (511.37,1801.19) .. controls (511.94,1798.9) and (513.36,1798.04) .. (515.65,1798.61) .. controls (517.94,1799.18) and (519.36,1798.32) .. (519.93,1796.03) .. controls (520.5,1793.74) and (521.92,1792.88) .. (524.21,1793.45) .. controls (526.5,1794.02) and (527.93,1793.16) .. (528.5,1790.87) .. controls (529.07,1788.58) and (530.49,1787.72) .. (532.78,1788.29) .. controls (535.07,1788.86) and (536.49,1788) .. (537.06,1785.71) .. controls (537.63,1783.42) and (539.06,1782.56) .. (541.35,1783.13) .. controls (543.64,1783.7) and (545.06,1782.84) .. (545.63,1780.55) .. controls (546.2,1778.26) and (547.62,1777.4) .. (549.91,1777.97) .. controls (552.2,1778.54) and (553.63,1777.68) .. (554.2,1775.39) .. controls (554.77,1773.1) and (556.19,1772.24) .. (558.48,1772.81) .. controls (560.77,1773.38) and (562.19,1772.52) .. (562.76,1770.23) .. controls (563.33,1767.94) and (564.76,1767.08) .. (567.05,1767.65) .. controls (569.34,1768.22) and (570.76,1767.36) .. (571.33,1765.07) .. controls (571.9,1762.78) and (573.32,1761.92) .. (575.61,1762.49) .. controls (577.9,1763.06) and (579.32,1762.2) .. (579.89,1759.91) .. controls (580.46,1757.62) and (581.89,1756.76) .. (584.18,1757.33) .. controls (586.47,1757.9) and (587.89,1757.04) .. (588.46,1754.75) .. controls (589.03,1752.46) and (590.45,1751.6) .. (592.74,1752.17) .. controls (595.03,1752.74) and (596.46,1751.88) .. (597.03,1749.59) .. controls (597.6,1747.3) and (599.02,1746.44) .. (601.31,1747.01) -- (601.33,1747) -- (601.33,1747) ;
\draw [shift={(540.93,1783.38)}, rotate = 148.94] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (0,5.59) -- (0,-5.59)(-5.03,5.59) -- (-5.03,-5.59)   ;
%Shape: Circle [id:dp31757419484776506] 
\draw   (458.58,1829.08) .. controls (458.58,1820.38) and (465.63,1813.33) .. (474.33,1813.33) .. controls (483.03,1813.33) and (490.08,1820.38) .. (490.08,1829.08) .. controls (490.08,1837.78) and (483.03,1844.83) .. (474.33,1844.83) .. controls (465.63,1844.83) and (458.58,1837.78) .. (458.58,1829.08) -- cycle ;
%Shape: Circle [id:dp7037723642891509] 
\draw   (601,1827.67) .. controls (601,1818.97) and (608.05,1811.92) .. (616.75,1811.92) .. controls (625.45,1811.92) and (632.5,1818.97) .. (632.5,1827.67) .. controls (632.5,1836.37) and (625.45,1843.42) .. (616.75,1843.42) .. controls (608.05,1843.42) and (601,1836.37) .. (601,1827.67) -- cycle ;
%Straight Lines [id:da6817621024152516] 
\draw    (615.75,1810.92) -- (615.25,1752) ;
\draw [shift={(615.46,1776.46)}, rotate = 89.51] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp6322825904226237] 
\draw   (561.67,319.83) .. controls (561.67,311.83) and (568.16,305.33) .. (576.17,305.33) .. controls (584.17,305.33) and (590.67,311.83) .. (590.67,319.83) .. controls (590.67,327.84) and (584.17,334.33) .. (576.17,334.33) .. controls (568.16,334.33) and (561.67,327.84) .. (561.67,319.83) -- cycle ;
%Shape: Rectangle [id:dp5917405752157332] 
\draw   (567.33,406) -- (585,406) -- (585,424.33) -- (567.33,424.33) -- cycle ;
%Shape: Rectangle [id:dp10293335087888145] 
\draw   (567.33,356) -- (585,356) -- (585,374.33) -- (567.33,374.33) -- cycle ;
%Straight Lines [id:da6621548108071567] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (577,405) -- (577,375.67) ;
\draw [shift={(577,385.33)}, rotate = 90] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6199767801112503] 
\draw    (576.17,355.17) -- (576.17,334.33) ;
\draw [shift={(576.17,339.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Rounded Rect [id:dp8666231534137869] 
\draw   (627.67,345.33) .. controls (627.67,338.71) and (633.04,333.33) .. (639.67,333.33) -- (675.67,333.33) .. controls (682.29,333.33) and (687.67,338.71) .. (687.67,345.33) -- (687.67,384.33) .. controls (687.67,390.96) and (682.29,396.33) .. (675.67,396.33) -- (639.67,396.33) .. controls (633.04,396.33) and (627.67,390.96) .. (627.67,384.33) -- cycle ;
%Shape: Circle [id:dp9762017403367442] 
\draw   (636.33,355) .. controls (636.33,348.74) and (641.41,343.67) .. (647.67,343.67) .. controls (653.93,343.67) and (659,348.74) .. (659,355) .. controls (659,361.26) and (653.93,366.33) .. (647.67,366.33) .. controls (641.41,366.33) and (636.33,361.26) .. (636.33,355) -- cycle ;
%Straight Lines [id:da693687151022812] 
\draw    (585.67,414.33) -- (637,388.33) ;
\draw [shift={(615.79,399.07)}, rotate = 153.14] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da13847713687249974] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (631.67,348.33) -- (590.33,327) ;
\draw [shift={(606.56,335.37)}, rotate = 27.3] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp3390618284080278] 
\draw   (656.33,375) .. controls (656.33,368.74) and (661.41,363.67) .. (667.67,363.67) .. controls (673.93,363.67) and (679,368.74) .. (679,375) .. controls (679,381.26) and (673.93,386.33) .. (667.67,386.33) .. controls (661.41,386.33) and (656.33,381.26) .. (656.33,375) -- cycle ;
%Shape: Circle [id:dp44759926780950066] 
\draw   (390,638.83) .. controls (390,630.83) and (396.49,624.33) .. (404.5,624.33) .. controls (412.51,624.33) and (419,630.83) .. (419,638.83) .. controls (419,646.84) and (412.51,653.33) .. (404.5,653.33) .. controls (396.49,653.33) and (390,646.84) .. (390,638.83) -- cycle ;
%Shape: Rectangle [id:dp02910146045205475] 
\draw   (469,675.67) -- (486.67,675.67) -- (486.67,694) -- (469,694) -- cycle ;
%Shape: Rectangle [id:dp7855584309373669] 
\draw   (397.67,675) -- (415.33,675) -- (415.33,693.33) -- (397.67,693.33) -- cycle ;
%Straight Lines [id:da2275837604754607] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (466.33,685) -- (415.67,685) ;
\draw [shift={(436,685)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da11383365769322684] 
\draw    (406.5,674.17) -- (406.5,653.33) ;
\draw [shift={(406.5,658.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da7842478675933815] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (458.67,638.83) -- (419,638.83) ;
\draw [shift={(433.83,638.83)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp639485771923263] 
\draw   (462,638.83) .. controls (462,630.83) and (468.49,624.33) .. (476.5,624.33) .. controls (484.51,624.33) and (491,630.83) .. (491,638.83) .. controls (491,646.84) and (484.51,653.33) .. (476.5,653.33) .. controls (468.49,653.33) and (462,646.84) .. (462,638.83) -- cycle ;
%Straight Lines [id:da10970431165424288] 
\draw    (476.33,675.67) -- (476.5,653.33) ;
\draw [shift={(476.45,659.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp5583364001063251] 
\draw   (541,638.67) -- (558.67,638.67) -- (558.67,657) -- (541,657) -- cycle ;
%Shape: Circle [id:dp9479151220592645] 
\draw   (534,601.83) .. controls (534,593.83) and (540.49,587.33) .. (548.5,587.33) .. controls (556.51,587.33) and (563,593.83) .. (563,601.83) .. controls (563,609.84) and (556.51,616.33) .. (548.5,616.33) .. controls (540.49,616.33) and (534,609.84) .. (534,601.83) -- cycle ;
%Straight Lines [id:da9641056705917042] 
\draw    (548.33,638.67) -- (548.5,616.33) ;
\draw [shift={(548.45,622.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp9612114366404982] 
\draw   (549,742.67) -- (566.67,742.67) -- (566.67,761) -- (549,761) -- cycle ;
%Shape: Circle [id:dp8075929649062006] 
\draw   (542,705.83) .. controls (542,697.83) and (548.49,691.33) .. (556.5,691.33) .. controls (564.51,691.33) and (571,697.83) .. (571,705.83) .. controls (571,713.84) and (564.51,720.33) .. (556.5,720.33) .. controls (548.49,720.33) and (542,713.84) .. (542,705.83) -- cycle ;
%Straight Lines [id:da652812436079679] 
\draw    (556.33,742.67) -- (556.5,720.33) ;
\draw [shift={(556.45,726.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp4808330623037198] 
\draw   (629,668.67) -- (646.67,668.67) -- (646.67,687) -- (629,687) -- cycle ;
%Shape: Circle [id:dp9803542080067535] 
\draw   (622,631.83) .. controls (622,623.83) and (628.49,617.33) .. (636.5,617.33) .. controls (644.51,617.33) and (651,623.83) .. (651,631.83) .. controls (651,639.84) and (644.51,646.33) .. (636.5,646.33) .. controls (628.49,646.33) and (622,639.84) .. (622,631.83) -- cycle ;
%Straight Lines [id:da5898621189097251] 
\draw    (636.33,668.67) -- (636.5,646.33) ;
\draw [shift={(636.45,652.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp7322784469286394] 
\draw   (429,773.67) -- (446.67,773.67) -- (446.67,792) -- (429,792) -- cycle ;
%Shape: Circle [id:dp03183378480384991] 
\draw   (422,736.83) .. controls (422,728.83) and (428.49,722.33) .. (436.5,722.33) .. controls (444.51,722.33) and (451,728.83) .. (451,736.83) .. controls (451,744.84) and (444.51,751.33) .. (436.5,751.33) .. controls (428.49,751.33) and (422,744.84) .. (422,736.83) -- cycle ;
%Straight Lines [id:da16477234358540804] 
\draw    (436.33,773.67) -- (436.5,751.33) ;
\draw [shift={(436.45,757.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da5493300774400265] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (534,607.83) -- (491,638.83) ;
\draw [shift={(508.44,626.26)}, rotate = 324.21] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da18500668267854858] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (622,631.83) -- (563,601.83) ;
\draw [shift={(588.04,614.57)}, rotate = 26.95] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da06712015849715303] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (622,633.83) -- (571,702.33) ;
\draw [shift={(593.51,672.09)}, rotate = 306.67] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da013119368423795574] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (465,653.33) -- (441,724.83) ;
\draw [shift={(451.41,693.82)}, rotate = 288.56] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da8923585368403464] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (476,698.33) -- (448,779.33) ;
\draw [shift={(460.37,743.56)}, rotate = 289.07] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da4468161562848043] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (548,752.33) -- (487,685.33) ;
\draw [shift={(514.13,715.14)}, rotate = 47.68] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da8300556253777622] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (628,677.33) -- (568,750.33) ;
\draw [shift={(594.83,717.7)}, rotate = 309.42] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da03303333297755118] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (539,699.33) -- (490,647.33) ;
\draw [shift={(511.07,669.69)}, rotate = 46.7] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da40874906846500414] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (540,648.33) -- (487,684.33) ;
\draw [shift={(509.36,669.14)}, rotate = 325.81] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da04836182888238194] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (628,677.33) -- (562.5,647.08) ;
\draw [shift={(590.71,660.11)}, rotate = 24.79] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da9794677594223615] 
\draw [line width=1.5]    (-10,296.5) -- (715,296.5) ;
%Shape: Ellipse [id:dp22538219164382167] 
\draw   (62.67,1720.5) .. controls (62.67,1697.58) and (88.11,1679) .. (119.5,1679) .. controls (150.89,1679) and (176.33,1697.58) .. (176.33,1720.5) .. controls (176.33,1743.42) and (150.89,1762) .. (119.5,1762) .. controls (88.11,1762) and (62.67,1743.42) .. (62.67,1720.5) -- cycle ;
%Shape: Rectangle [id:dp6597543047687238] 
\draw   (84,1696.67) -- (154,1696.67) -- (154,1736.67) -- (84,1736.67) -- cycle ;
%Straight Lines [id:da7308175760583218] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=1.5]    (142.67,1718.33) -- (232.67,1808.33) ;
\draw [shift={(192.47,1768.14)}, rotate = 225] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (11.61,-5.58) -- (0,0) -- (11.61,5.58) -- cycle    ;
%Shape: Rectangle [id:dp44697551504851574] 
\draw   (212,1791.5) -- (253.33,1791.5) -- (253.33,1825.17) -- (212,1825.17) -- cycle ;
%Straight Lines [id:da8090409070265012] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=1.5]  [dash pattern={on 5.63pt off 4.5pt}]  (201.67,1667.33) -- (274.67,1733.33) ;
\draw [shift={(243.21,1704.89)}, rotate = 222.12] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (11.61,-5.58) -- (0,0) -- (11.61,5.58) -- cycle    ;
%Shape: Ellipse [id:dp4058818642919557] 
\draw   (192.33,1810) .. controls (192.33,1792.88) and (209.35,1779) .. (230.33,1779) .. controls (251.32,1779) and (268.33,1792.88) .. (268.33,1810) .. controls (268.33,1827.12) and (251.32,1841) .. (230.33,1841) .. controls (209.35,1841) and (192.33,1827.12) .. (192.33,1810) -- cycle ;
%Shape: Circle [id:dp96744446424547] 
\draw   (173.17,1660.58) .. controls (173.17,1651.88) and (180.22,1644.83) .. (188.92,1644.83) .. controls (197.62,1644.83) and (204.67,1651.88) .. (204.67,1660.58) .. controls (204.67,1669.28) and (197.62,1676.33) .. (188.92,1676.33) .. controls (180.22,1676.33) and (173.17,1669.28) .. (173.17,1660.58) -- cycle ;
%Shape: Circle [id:dp2830340677047376] 
\draw   (271.17,1740.58) .. controls (271.17,1731.88) and (278.22,1724.83) .. (286.92,1724.83) .. controls (295.62,1724.83) and (302.67,1731.88) .. (302.67,1740.58) .. controls (302.67,1749.28) and (295.62,1756.33) .. (286.92,1756.33) .. controls (278.22,1756.33) and (271.17,1749.28) .. (271.17,1740.58) -- cycle ;
%Straight Lines [id:da9105050967920671] 
\draw [line width=0.75]    (142.67,1718.33) -- (178.67,1673.33) ;
\draw [shift={(163.79,1691.93)}, rotate = 128.66] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da05112918301286684] 
\draw    (232.67,1808.33) -- (276.67,1753.33) ;
\draw [shift={(257.79,1776.93)}, rotate = 128.66] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da2619241061858497] 
\draw [line width=0.75]    (142.67,1718.33) .. controls (144.6,1716.98) and (146.24,1717.26) .. (147.59,1719.19) .. controls (148.95,1721.12) and (150.59,1721.4) .. (152.52,1720.04) .. controls (154.45,1718.68) and (156.09,1718.96) .. (157.45,1720.89) .. controls (158.8,1722.82) and (160.44,1723.1) .. (162.37,1721.75) .. controls (164.3,1720.39) and (165.94,1720.67) .. (167.3,1722.6) .. controls (168.66,1724.53) and (170.3,1724.81) .. (172.23,1723.45) .. controls (174.16,1722.09) and (175.8,1722.37) .. (177.15,1724.3) .. controls (178.51,1726.23) and (180.15,1726.51) .. (182.08,1725.16) .. controls (184.01,1723.8) and (185.65,1724.08) .. (187.01,1726.01) .. controls (188.36,1727.94) and (190,1728.22) .. (191.93,1726.86) .. controls (193.86,1725.51) and (195.5,1725.79) .. (196.86,1727.72) .. controls (198.22,1729.65) and (199.86,1729.93) .. (201.79,1728.57) .. controls (203.72,1727.21) and (205.36,1727.49) .. (206.71,1729.42) .. controls (208.07,1731.35) and (209.71,1731.63) .. (211.64,1730.28) .. controls (213.57,1728.92) and (215.21,1729.2) .. (216.57,1731.13) .. controls (217.92,1733.06) and (219.56,1733.34) .. (221.49,1731.98) .. controls (223.42,1730.63) and (225.06,1730.91) .. (226.42,1732.84) .. controls (227.78,1734.77) and (229.42,1735.05) .. (231.35,1733.69) .. controls (233.28,1732.33) and (234.92,1732.61) .. (236.27,1734.54) .. controls (237.63,1736.47) and (239.27,1736.75) .. (241.2,1735.39) .. controls (243.13,1734.04) and (244.77,1734.32) .. (246.13,1736.25) .. controls (247.48,1738.18) and (249.12,1738.46) .. (251.05,1737.1) .. controls (252.98,1735.74) and (254.62,1736.02) .. (255.98,1737.95) .. controls (257.34,1739.88) and (258.98,1740.16) .. (260.91,1738.81) .. controls (262.84,1737.45) and (264.48,1737.73) .. (265.83,1739.66) .. controls (267.19,1741.59) and (268.83,1741.87) .. (270.76,1740.51) -- (271.17,1740.58) -- (271.17,1740.58) ;
\draw [shift={(203.96,1728.95)}, rotate = 189.82] [color={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.75]    (0,5.59) -- (0,-5.59)(-5.03,5.59) -- (-5.03,-5.59)   ;
%Shape: Circle [id:dp8584718210028095] 
\draw   (30.67,1794.83) .. controls (30.67,1786.83) and (37.16,1780.33) .. (45.17,1780.33) .. controls (53.17,1780.33) and (59.67,1786.83) .. (59.67,1794.83) .. controls (59.67,1802.84) and (53.17,1809.33) .. (45.17,1809.33) .. controls (37.16,1809.33) and (30.67,1802.84) .. (30.67,1794.83) -- cycle ;
%Shape: Rectangle [id:dp7826369183950601] 
\draw   (109.67,1831.67) -- (127.33,1831.67) -- (127.33,1850) -- (109.67,1850) -- cycle ;
%Shape: Rectangle [id:dp7921124516756475] 
\draw   (38.33,1831) -- (56,1831) -- (56,1849.33) -- (38.33,1849.33) -- cycle ;
%Straight Lines [id:da451090984933604] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (107,1841) -- (56.33,1841) ;
\draw [shift={(76.67,1841)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6171879953417931] 
\draw    (47.17,1830.17) -- (47.17,1809.33) ;
\draw [shift={(47.17,1814.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da06568825136171896] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (101.67,1794) -- (62,1794) ;
\draw [shift={(76.83,1794)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp9894970001752357] 
\draw   (102.67,1794.83) .. controls (102.67,1786.83) and (109.16,1780.33) .. (117.17,1780.33) .. controls (125.17,1780.33) and (131.67,1786.83) .. (131.67,1794.83) .. controls (131.67,1802.84) and (125.17,1809.33) .. (117.17,1809.33) .. controls (109.16,1809.33) and (102.67,1802.84) .. (102.67,1794.83) -- cycle ;
%Straight Lines [id:da7228476202938143] 
\draw    (117,1831.67) -- (117.17,1809.33) ;
\draw [shift={(117.12,1815.5)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp7385182061700386] 
\draw   (53.33,331.17) .. controls (53.33,323.16) and (59.83,316.67) .. (67.83,316.67) .. controls (75.84,316.67) and (82.33,323.16) .. (82.33,331.17) .. controls (82.33,339.17) and (75.84,345.67) .. (67.83,345.67) .. controls (59.83,345.67) and (53.33,339.17) .. (53.33,331.17) -- cycle ;
%Shape: Triangle [id:dp4534580989914354] 
\draw   (112,485) -- (147,525) -- (77,525) -- cycle ;
%Shape: Triangle [id:dp40034981789248114] 
\draw   (32.67,485) -- (67.67,525) -- (-2.33,525) -- cycle ;
%Straight Lines [id:da13750305090053705] 
\draw    (32.67,485) -- (68,463) ;
%Straight Lines [id:da7861911710602139] 
\draw    (112,485) -- (68,463) ;
%Shape: Rectangle [id:dp5060545708912132] 
\draw   (59,417.33) -- (76.67,417.33) -- (76.67,435.67) -- (59,435.67) -- cycle ;
%Shape: Rectangle [id:dp20340400788237956] 
\draw   (59,367.33) -- (76.67,367.33) -- (76.67,385.67) -- (59,385.67) -- cycle ;
%Straight Lines [id:da1335919609660845] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ]   (68.67,416.33) -- (68.67,387) ;
\draw [shift={(68.67,401.67)}, rotate = 90] [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=0.75]    (0,5.59) -- (0,-5.59)   ;
%Straight Lines [id:da6329269055971283] 
\draw    (68,463) -- (68.67,436.33) ;
\draw [shift={(68.46,444.67)}, rotate = 91.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da8082502255517092] 
\draw    (68.67,366.83) -- (68.67,346) ;
\draw [shift={(68.67,351.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp9376226202116167] 
\draw   (171.67,331.83) .. controls (171.67,323.83) and (178.16,317.33) .. (186.17,317.33) .. controls (194.17,317.33) and (200.67,323.83) .. (200.67,331.83) .. controls (200.67,339.84) and (194.17,346.33) .. (186.17,346.33) .. controls (178.16,346.33) and (171.67,339.84) .. (171.67,331.83) -- cycle ;
%Shape: Rectangle [id:dp9432523025199191] 
\draw   (177.33,418) -- (195,418) -- (195,436.33) -- (177.33,436.33) -- cycle ;
%Shape: Rectangle [id:dp05207872956448667] 
\draw   (177.33,368) -- (195,368) -- (195,386.33) -- (177.33,386.33) -- cycle ;
%Straight Lines [id:da3087103009059926] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (187,417) -- (187,387.67) ;
\draw [shift={(187,397.33)}, rotate = 90] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da632819876225664] 
\draw    (186.17,367.17) -- (186.17,346.33) ;
\draw [shift={(186.17,351.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Rounded Rect [id:dp3566091909543969] 
\draw   (237.67,357.33) .. controls (237.67,350.71) and (243.04,345.33) .. (249.67,345.33) -- (285.67,345.33) .. controls (292.29,345.33) and (297.67,350.71) .. (297.67,357.33) -- (297.67,396.33) .. controls (297.67,402.96) and (292.29,408.33) .. (285.67,408.33) -- (249.67,408.33) .. controls (243.04,408.33) and (237.67,402.96) .. (237.67,396.33) -- cycle ;
%Shape: Circle [id:dp6438985306816862] 
\draw   (246.33,367) .. controls (246.33,360.74) and (251.41,355.67) .. (257.67,355.67) .. controls (263.93,355.67) and (269,360.74) .. (269,367) .. controls (269,373.26) and (263.93,378.33) .. (257.67,378.33) .. controls (251.41,378.33) and (246.33,373.26) .. (246.33,367) -- cycle ;
%Straight Lines [id:da5974725966511654] 
\draw    (195.67,426.33) -- (247,400.33) ;
\draw [shift={(225.79,411.07)}, rotate = 153.14] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da4393234608339813] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (241.67,360.33) -- (200.33,339) ;
\draw [shift={(216.56,347.37)}, rotate = 27.3] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp977160224056534] 
\draw   (266.33,387) .. controls (266.33,380.74) and (271.41,375.67) .. (277.67,375.67) .. controls (283.93,375.67) and (289,380.74) .. (289,387) .. controls (289,393.26) and (283.93,398.33) .. (277.67,398.33) .. controls (271.41,398.33) and (266.33,393.26) .. (266.33,387) -- cycle ;
%Shape: Circle [id:dp5163064070443517] 
\draw   (183,473.5) .. controls (183,465.49) and (189.49,459) .. (197.5,459) .. controls (205.51,459) and (212,465.49) .. (212,473.5) .. controls (212,481.51) and (205.51,488) .. (197.5,488) .. controls (189.49,488) and (183,481.51) .. (183,473.5) -- cycle ;
%Shape: Rectangle [id:dp06683772015092693] 
\draw   (262,510.33) -- (279.67,510.33) -- (279.67,528.67) -- (262,528.67) -- cycle ;
%Shape: Rectangle [id:dp2096447500441847] 
\draw   (190.67,509.67) -- (208.33,509.67) -- (208.33,528) -- (190.67,528) -- cycle ;
%Straight Lines [id:da8960241981963313] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (259.33,519.67) -- (208.67,519.67) ;
\draw [shift={(229,519.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6042482007669181] 
\draw    (199.5,508.83) -- (199.5,488) ;
\draw [shift={(199.5,493.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da22733512794963184] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (254,472.67) -- (214.33,472.67) ;
\draw [shift={(229.17,472.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp5816613679729856] 
\draw   (255,473.5) .. controls (255,465.49) and (261.49,459) .. (269.5,459) .. controls (277.51,459) and (284,465.49) .. (284,473.5) .. controls (284,481.51) and (277.51,488) .. (269.5,488) .. controls (261.49,488) and (255,481.51) .. (255,473.5) -- cycle ;
%Straight Lines [id:da99489776063394] 
\draw    (269.33,510.33) -- (269.5,488) ;
\draw [shift={(269.45,494.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6166945279253642] 
\draw [line width=1.5]    (-11,821) -- (714,821) ;
%Shape: Rectangle [id:dp8987992222410803] 
\draw   (464.08,2132.23) -- (512.43,2132.23) -- (512.43,2171.66) -- (464.08,2171.66) -- cycle ;
%Shape: Rectangle [id:dp06192674638934781] 
\draw   (444.36,2046.31) -- (492.72,2046.31) -- (492.72,2085.75) -- (444.36,2085.75) -- cycle ;
%Shape: Ellipse [id:dp7110377996200195] 
\draw   (448.82,2060.93) .. controls (448.82,2054.41) and (455.29,2049.13) .. (463.28,2049.13) .. controls (471.26,2049.13) and (477.74,2054.41) .. (477.74,2060.93) .. controls (477.74,2067.44) and (471.26,2072.72) .. (463.28,2072.72) .. controls (455.29,2072.72) and (448.82,2067.44) .. (448.82,2060.93) -- cycle ;
%Shape: Rectangle [id:dp695782079197981] 
\draw   (552.1,2045.61) -- (600.46,2045.61) -- (600.46,2085.04) -- (552.1,2085.04) -- cycle ;
%Shape: Ellipse [id:dp18312013102374958] 
\draw   (556.56,2060.22) .. controls (556.56,2053.71) and (563.04,2048.43) .. (571.02,2048.43) .. controls (579.01,2048.43) and (585.48,2053.71) .. (585.48,2060.22) .. controls (585.48,2066.74) and (579.01,2072.02) .. (571.02,2072.02) .. controls (563.04,2072.02) and (556.56,2066.74) .. (556.56,2060.22) -- cycle ;
%Shape: Rectangle [id:dp1991172647233619] 
\draw   (374.64,1961.81) -- (423,1961.81) -- (423,2001.24) -- (374.64,2001.24) -- cycle ;
%Shape: Ellipse [id:dp07342100833576182] 
\draw   (379.1,1976.42) .. controls (379.1,1969.91) and (385.58,1964.62) .. (393.56,1964.62) .. controls (401.55,1964.62) and (408.02,1969.91) .. (408.02,1976.42) .. controls (408.02,1982.93) and (401.55,1988.22) .. (393.56,1988.22) .. controls (385.58,1988.22) and (379.1,1982.93) .. (379.1,1976.42) -- cycle ;
%Shape: Rectangle [id:dp6674794605589673] 
\draw   (480.27,1958.99) -- (528.63,1958.99) -- (528.63,1998.43) -- (480.27,1998.43) -- cycle ;
%Shape: Ellipse [id:dp7436976462903653] 
\draw   (484.73,1973.6) .. controls (484.73,1967.09) and (491.21,1961.81) .. (499.19,1961.81) .. controls (507.18,1961.81) and (513.65,1967.09) .. (513.65,1973.6) .. controls (513.65,1980.12) and (507.18,1985.4) .. (499.19,1985.4) .. controls (491.21,1985.4) and (484.73,1980.12) .. (484.73,1973.6) -- cycle ;
%Shape: Rectangle [id:dp5725854067894174] 
\draw   (579.57,1963.22) -- (627.92,1963.22) -- (627.92,2002.65) -- (579.57,2002.65) -- cycle ;
%Shape: Ellipse [id:dp5674596716181819] 
\draw   (584.03,1977.83) .. controls (584.03,1971.31) and (590.5,1966.03) .. (598.49,1966.03) .. controls (606.47,1966.03) and (612.95,1971.31) .. (612.95,1977.83) .. controls (612.95,1984.34) and (606.47,1989.62) .. (598.49,1989.62) .. controls (590.5,1989.62) and (584.03,1984.34) .. (584.03,1977.83) -- cycle ;
%Shape: Rectangle [id:dp10508013945847616] 
\draw   (551.4,2131.52) -- (599.76,2131.52) -- (599.76,2170.96) -- (551.4,2170.96) -- cycle ;
%Shape: Rectangle [id:dp3556340712403967] 
\draw   (643.65,2075.89) -- (692.01,2075.89) -- (692.01,2115.33) -- (643.65,2115.33) -- cycle ;
%Shape: Ellipse [id:dp5057887735731792] 
\draw   (648.11,2090.5) .. controls (648.11,2083.99) and (654.58,2078.71) .. (662.57,2078.71) .. controls (670.56,2078.71) and (677.03,2083.99) .. (677.03,2090.5) .. controls (677.03,2097.02) and (670.56,2102.3) .. (662.57,2102.3) .. controls (654.58,2102.3) and (648.11,2097.02) .. (648.11,2090.5) -- cycle ;
%Straight Lines [id:da261138033427571] 
\draw    (463.28,2060.93) -- (483,2146.84) ;
\draw [shift={(474.26,2108.76)}, rotate = 257.07] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da1255241499758788] 
\draw    (400.6,1972.19) -- (483.56,2054.06) ;
\draw [shift={(445.64,2016.64)}, rotate = 224.62] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da07220762247683421] 
\draw    (392.01,1980.12) -- (463.28,2060.93) ;
\draw [shift={(430.95,2024.27)}, rotate = 228.59] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da27171009397847623] 
\draw    (662.57,2090.5) -- (570.32,2146.13) ;
\draw [shift={(612.16,2120.9)}, rotate = 328.91] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da11962321839646561] 
\draw    (598.49,1977.83) -- (571.02,2060.22) ;
\draw [shift={(583.17,2023.77)}, rotate = 288.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da18664580064298586] 
\draw    (571.02,2060.22) -- (570.32,2146.13) ;
\draw [shift={(570.63,2108.18)}, rotate = 270.47] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da3016201836895087] 
\draw    (499.19,1973.6) -- (571.02,2060.22) ;
\draw [shift={(538.3,2020.76)}, rotate = 230.33] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da2684828221199864] 
\draw    (425.59,1972.31) -- (478.74,1972.31) ;
\draw [shift={(481.74,1972.31)}, rotate = 180] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
\draw [shift={(422.59,1972.31)}, rotate = 0] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Rectangle [id:dp38325159790374164] 
\draw   (461.23,1850.66) -- (526.81,1850.66) -- (526.81,1890.98) -- (461.23,1890.98) -- cycle ;
%Shape: Ellipse [id:dp4888384768330196] 
\draw   (470.95,1870.91) .. controls (470.95,1864.1) and (476.83,1858.58) .. (484.09,1858.58) .. controls (491.35,1858.58) and (497.24,1864.1) .. (497.24,1870.91) .. controls (497.24,1877.71) and (491.35,1883.23) .. (484.09,1883.23) .. controls (476.83,1883.23) and (470.95,1877.71) .. (470.95,1870.91) -- cycle ;
%Straight Lines [id:da14528730234043197] 
\draw    (490.19,1869.15) -- (500.19,1974.6) ;
\draw [shift={(495.67,1926.85)}, rotate = 264.58] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da6175689427434128] 
\draw    (476.81,1872.67) -- (392.54,1968.79) ;
\draw [shift={(431.38,1924.49)}, rotate = 311.24] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Ellipse [id:dp40829048476842833] 
\draw   (484.09,1870.91) .. controls (484.09,1864.1) and (489.98,1858.58) .. (497.24,1858.58) .. controls (504.5,1858.58) and (510.38,1864.1) .. (510.38,1870.91) .. controls (510.38,1877.71) and (504.5,1883.23) .. (497.24,1883.23) .. controls (489.98,1883.23) and (484.09,1877.71) .. (484.09,1870.91) -- cycle ;
%Shape: Ellipse [id:dp3010350482390538] 
\draw   (526.03,884.18) .. controls (535.93,872.32) and (562.79,878.44) .. (586.02,897.85) .. controls (609.24,917.25) and (620.04,942.59) .. (610.13,954.45) .. controls (600.23,966.3) and (573.37,960.18) .. (550.14,940.78) .. controls (526.92,921.37) and (516.12,896.03) .. (526.03,884.18) -- cycle ;
%Shape: Ellipse [id:dp18444171034501533] 
\draw   (572.84,950.65) .. controls (563.73,938.18) and (576.14,913.59) .. (600.57,895.73) .. controls (625,877.87) and (652.2,873.5) .. (661.32,885.97) .. controls (670.44,898.44) and (658.02,923.03) .. (633.59,940.89) .. controls (609.16,958.76) and (581.96,963.13) .. (572.84,950.65) -- cycle ;
%Straight Lines [id:da5851147121314799] 
\draw    (631.5,893) -- (553.5,893) ;
%Straight Lines [id:da15681195757443867] 
\draw [line width=1.5]    (-15,1134.5) -- (710,1134.5) ;
%Shape: Circle [id:dp852112780233941] 
\draw   (98.33,1350.17) .. controls (98.33,1342.16) and (104.83,1335.67) .. (112.83,1335.67) .. controls (120.84,1335.67) and (127.33,1342.16) .. (127.33,1350.17) .. controls (127.33,1358.17) and (120.84,1364.67) .. (112.83,1364.67) .. controls (104.83,1364.67) and (98.33,1358.17) .. (98.33,1350.17) -- cycle ;
%Shape: Triangle [id:dp7924587749873859] 
\draw   (157,1504) -- (192,1544) -- (122,1544) -- cycle ;
%Shape: Triangle [id:dp6933711327042797] 
\draw   (77.67,1504) -- (112.67,1544) -- (42.67,1544) -- cycle ;
%Straight Lines [id:da5413618021571207] 
\draw    (77.67,1504) -- (113,1482) ;
%Straight Lines [id:da36089198638567943] 
\draw    (157,1504) -- (113,1482) ;
%Shape: Rectangle [id:dp2271584183083888] 
\draw   (104,1436.33) -- (121.67,1436.33) -- (121.67,1454.67) -- (104,1454.67) -- cycle ;
%Shape: Rectangle [id:dp954040833967732] 
\draw   (104,1386.33) -- (121.67,1386.33) -- (121.67,1404.67) -- (104,1404.67) -- cycle ;
%Straight Lines [id:da5156888114856979] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ]   (113.67,1435.33) -- (113.67,1406) ;
\draw [shift={(113.67,1420.67)}, rotate = 90] [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][line width=0.75]    (0,5.59) -- (0,-5.59)   ;
%Straight Lines [id:da7661574725552134] 
\draw    (113,1482) -- (113.67,1455.33) ;
\draw [shift={(113.46,1463.67)}, rotate = 91.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da9112848832430454] 
\draw    (113.67,1385.83) -- (113.67,1365) ;
\draw [shift={(113.67,1370.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp09456332838456105] 
\draw   (220.67,1338.83) .. controls (220.67,1330.83) and (227.16,1324.33) .. (235.17,1324.33) .. controls (243.17,1324.33) and (249.67,1330.83) .. (249.67,1338.83) .. controls (249.67,1346.84) and (243.17,1353.33) .. (235.17,1353.33) .. controls (227.16,1353.33) and (220.67,1346.84) .. (220.67,1338.83) -- cycle ;
%Shape: Rectangle [id:dp9475478917294664] 
\draw   (226.33,1425) -- (244,1425) -- (244,1443.33) -- (226.33,1443.33) -- cycle ;
%Shape: Rectangle [id:dp6053728388486752] 
\draw   (226.33,1375) -- (244,1375) -- (244,1393.33) -- (226.33,1393.33) -- cycle ;
%Straight Lines [id:da05370287868667556] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (236,1424) -- (236,1394.67) ;
\draw [shift={(236,1404.33)}, rotate = 90] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da8271787485709368] 
\draw    (235.17,1374.17) -- (235.17,1353.33) ;
\draw [shift={(235.17,1358.75)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Rounded Rect [id:dp8443218499225029] 
\draw   (286.67,1364.33) .. controls (286.67,1357.71) and (292.04,1352.33) .. (298.67,1352.33) -- (334.67,1352.33) .. controls (341.29,1352.33) and (346.67,1357.71) .. (346.67,1364.33) -- (346.67,1403.33) .. controls (346.67,1409.96) and (341.29,1415.33) .. (334.67,1415.33) -- (298.67,1415.33) .. controls (292.04,1415.33) and (286.67,1409.96) .. (286.67,1403.33) -- cycle ;
%Shape: Circle [id:dp08648039297093724] 
\draw   (295.33,1374) .. controls (295.33,1367.74) and (300.41,1362.67) .. (306.67,1362.67) .. controls (312.93,1362.67) and (318,1367.74) .. (318,1374) .. controls (318,1380.26) and (312.93,1385.33) .. (306.67,1385.33) .. controls (300.41,1385.33) and (295.33,1380.26) .. (295.33,1374) -- cycle ;
%Straight Lines [id:da20361244403733814] 
\draw    (244.67,1433.33) -- (296,1407.33) ;
\draw [shift={(274.79,1418.07)}, rotate = 153.14] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da11245837759343136] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (290.67,1367.33) -- (249.33,1346) ;
\draw [shift={(265.56,1354.37)}, rotate = 27.3] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp30366246654220075] 
\draw   (315.33,1394) .. controls (315.33,1387.74) and (320.41,1382.67) .. (326.67,1382.67) .. controls (332.93,1382.67) and (338,1387.74) .. (338,1394) .. controls (338,1400.26) and (332.93,1405.33) .. (326.67,1405.33) .. controls (320.41,1405.33) and (315.33,1400.26) .. (315.33,1394) -- cycle ;
%Shape: Circle [id:dp3170617820508006] 
\draw   (232,1480.5) .. controls (232,1472.49) and (238.49,1466) .. (246.5,1466) .. controls (254.51,1466) and (261,1472.49) .. (261,1480.5) .. controls (261,1488.51) and (254.51,1495) .. (246.5,1495) .. controls (238.49,1495) and (232,1488.51) .. (232,1480.5) -- cycle ;
%Shape: Rectangle [id:dp031664500269476425] 
\draw   (311,1517.33) -- (328.67,1517.33) -- (328.67,1535.67) -- (311,1535.67) -- cycle ;
%Shape: Rectangle [id:dp5947708792284003] 
\draw   (239.67,1516.67) -- (257.33,1516.67) -- (257.33,1535) -- (239.67,1535) -- cycle ;
%Straight Lines [id:da8589271659641553] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ][fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ]   (308.33,1526.67) -- (257.67,1526.67) ;
\draw [shift={(278,1526.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da23286801215246555] 
\draw    (248.5,1515.83) -- (248.5,1495) ;
\draw [shift={(248.5,1500.42)}, rotate = 90] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Straight Lines [id:da3636688875968528] 
\draw [color={rgb, 255:red, 208; green, 2; blue, 27 }  ,draw opacity=1 ] [dash pattern={on 4.5pt off 4.5pt}]  (303,1479.67) -- (263.33,1479.67) ;
\draw [shift={(278.17,1479.67)}, rotate = 360] [fill={rgb, 255:red, 208; green, 2; blue, 27 }  ,fill opacity=1 ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;
%Shape: Circle [id:dp06836145697941864] 
\draw   (304,1480.5) .. controls (304,1472.49) and (310.49,1466) .. (318.5,1466) .. controls (326.51,1466) and (333,1472.49) .. (333,1480.5) .. controls (333,1488.51) and (326.51,1495) .. (318.5,1495) .. controls (310.49,1495) and (304,1488.51) .. (304,1480.5) -- cycle ;
%Straight Lines [id:da5017332792794225] 
\draw    (318.33,1517.33) -- (318.5,1495) ;
\draw [shift={(318.45,1501.17)}, rotate = 90.43] [fill={rgb, 255:red, 0; green, 0; blue, 0 }  ][line width=0.08]  [draw opacity=0] (8.93,-4.29) -- (0,0) -- (8.93,4.29) -- cycle    ;

% Text Node
\draw (220,37.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$f:\mathtt{t\rightarrow o}$};
% Text Node
\draw (223.33,136.73) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$g:\mathtt{s\rightarrow o} ,\forall g\in G[ s]$};
% Text Node
\draw (81,41) node    {$f$};
% Text Node
\draw (117,213.73) node [anchor=north west][inner sep=0.75pt]    {$s_{n}$};
% Text Node
\draw (77,79.73) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (37,214.4) node [anchor=north west][inner sep=0.75pt]    {$s_{1}$};
% Text Node
\draw (76.33,129.73) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (198.17,43.83) node    {$f$};
% Text Node
\draw (195.33,80.4) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (194.67,130.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (269.67,79) node  [font=\scriptsize]  {$g_{i}$};
% Text Node
\draw (298,43.33) node  [font=\footnotesize]  {$G[ s]$};
% Text Node
\draw (289.67,99) node  [font=\scriptsize]  {$g_{n}$};
% Text Node
\draw (209.5,185.5) node    {$f$};
% Text Node
\draw (208.67,222.07) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (278.67,223.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (281.5,185.5) node    {$g_{i}$};
% Text Node
\draw (235.33,163.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{f,g_{i}}$};
% Text Node
\draw (46,259) node [anchor=north west][inner sep=0.75pt]   [align=left] {Parse Tree};
% Text Node
\draw (441,258) node [anchor=north west][inner sep=0.75pt]   [align=left] {Metaphor Graph (Global)};
% Text Node
\draw (385.5,69.5) node    {$f_{4}$};
% Text Node
\draw (381.67,106.07) node [anchor=north west][inner sep=0.75pt]    {$t_{4}$};
% Text Node
\draw (452,105.73) node [anchor=north west][inner sep=0.75pt]    {$t_{3}$};
% Text Node
\draw (457.5,69.5) node    {$f_{3}$};
% Text Node
\draw (411.33,51.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{3,4}$};
% Text Node
\draw (525,69.73) node [anchor=north west][inner sep=0.75pt]    {$t_{1}$};
% Text Node
\draw (529.5,32.5) node    {$f_{1}$};
% Text Node
\draw (532,172.73) node [anchor=north west][inner sep=0.75pt]    {$t_{2}$};
% Text Node
\draw (537.5,136.5) node    {$f_{2}$};
% Text Node
\draw (613,99.73) node [anchor=north west][inner sep=0.75pt]    {$t_{q}$};
% Text Node
\draw (617.5,62.5) node    {$f_{q}$};
% Text Node
\draw (412,203.73) node [anchor=north west][inner sep=0.75pt]    {$t_{5}$};
% Text Node
\draw (417.5,167.5) node    {$f_{5}$};
% Text Node
\draw (194,259) node [anchor=north west][inner sep=0.75pt]   [align=left] {Metaphor (Local)};
% Text Node
\draw (570.33,25.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{q,1}$};
% Text Node
\draw (471.33,33.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{1,3}$};
% Text Node
\draw (396.33,133.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{3,5}$};
% Text Node
\draw (445,216.4) node [anchor=north west][inner sep=0.75pt]    {$P_{j}( q) =\max_{i,j}( w_{i,j} \cdot u_{i,j} \cdot P_{i}( q))$};
% Text Node
\draw (10,108.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$inconsistent$};
% Text Node
\draw (467.33,1725.73) node [anchor=north west][inner sep=0.75pt]    {$f_{j}$};
% Text Node
\draw (609.67,1727.73) node [anchor=north west][inner sep=0.75pt]    {$f_{i}$};
% Text Node
\draw (509.67,1700.73) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$c_{j,i} :F_{j}\rightarrow F_{i}$};
% Text Node
\draw (521,1804.4) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$c_{j,i} :S_{j}\rightarrow S_{i}$};
% Text Node
\draw (468.67,1821.07) node [anchor=north west][inner sep=0.75pt]    {$s_{j}$};
% Text Node
\draw (610.67,1820.07) node [anchor=north west][inner sep=0.75pt]    {$s_{i}$};
% Text Node
\draw (10,627.07) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
1.\ \mathrm{dfs}( q,t) ,\ \mathsf{delete\ the\ backward\ edges\ to\ form\ a\ dag}\\
2.\ \mathsf{topo\ sort\ the\ nodes\ to\ form\ structure} \ p_{q} =1\\
3.\ p_{j} \ =\ \max_{i,j}( p_{i} \cdot w_{i,j} \cdot u_{i,j}) ,\ \\
\ \ \ \ s_{j} =\max_{i,j}[ c_{i,j}( s_{i})][\mathtt{key} =p_{i} \cdot w_{i,j} \cdot u_{i,j}]
\end{array}$};
% Text Node
\draw (12,710.4) node [anchor=north west][inner sep=0.75pt]    {$4.\ p_{j} \ \leftarrow p_{j} \ \prod _{j,k}( 1-w_{j,k} \cdot u_{j,k}) \ update\ after\ ( 3) .$};
% Text Node
\draw (13,748.4) node [anchor=north west][inner sep=0.75pt]    {$5.\ \mathbb{E}[ f_{q}( s_{q})] \ =\ \sum _{i} p_{i}( q) f_{i}( s_{i})$};
% Text Node
\draw (9,599.9) node [anchor=north west][inner sep=0.75pt]    {$\mathrm{Evaluate( f_{q} ,\ s_{q})}$};
% Text Node
\draw (334,318.9) node [anchor=north west][inner sep=0.75pt]    {$\mathrm{InferMetaphor( f,\ s)}$};
% Text Node
\draw (334,347.9) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
\mathsf{Input:\ function} \ f,\mathtt{\ arguments} \ s\\
1.\ t\ =\ \mathsf{input\_type} \ [ f]\\
\ \ \ \ \mathtt{if} \ t\ ==\ s:\ \mathtt{return} \ \emptyset \\
2.\ \Pr\left[ f,\ s\rightarrow t\right] \ \mathsf{is\ calculated}\\
3.\ \Pr\left[ f,\ s\rightarrow t\right] \ < \ \theta \ \rightarrow \ \mathtt{add\ metaphor}\\
\ \ \ \ \ \mathtt{infer\ location}\Pr\left[ f,\ s\rightarrow t\right] \ \mathtt{as} \ h\ =\ f\\
\ \ \ \ \ \mathsf{dfs\ to\ find\ the\ most\ probable\ node\ to\ start}\\
4.\ \mathsf{add\ the\ connection} \ \forall g\in G[ s] ,\ c_{g,f} \ \mathsf{as\ the\ caster\ }\\
\mathsf{\ \ \ \ neural\ network\ ( same\ \forall g) \ and\ } w_{g,f}\\
\ \ \ \ c_{g,f}( s_{g}) \ =\ s_{f} ,\ u_{f} \ \mathsf{where\ } u_{f} \in [ 0,1] ,\ s_{f} \ \mathtt{is\ value}
\end{array}$};
% Text Node
\draw (409,1597.4) node [anchor=north west][inner sep=0.75pt]    {$Curriculum\ to\ gradually\ learn\ the\ $};
% Text Node
\draw (443,1682.4) node [anchor=north west][inner sep=0.75pt]    {$f_{j}( s_{j}) =\ f_{j} \circ c_{j,i}( s_{i}) \ =\ f_{i}( s_{i}) \approx p_{j}$};
% Text Node
\draw (598,313.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$f:\mathtt{t\rightarrow o}$};
% Text Node
\draw (601.33,412.73) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$g:\mathtt{s\rightarrow o} ,\forall g\in G[ s]$};
% Text Node
\draw (576.17,319.83) node    {$f$};
% Text Node
\draw (573.33,356.4) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (572.67,406.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (647.67,355) node  [font=\scriptsize]  {$g_{i}$};
% Text Node
\draw (676,319.33) node  [font=\footnotesize]  {$G[ s]$};
% Text Node
\draw (667.67,375) node  [font=\scriptsize]  {$g_{n}$};
% Text Node
\draw (404.5,638.83) node    {$f_{4}$};
% Text Node
\draw (400.67,675.4) node [anchor=north west][inner sep=0.75pt]    {$t_{4}$};
% Text Node
\draw (471,675.07) node [anchor=north west][inner sep=0.75pt]    {$t_{3}$};
% Text Node
\draw (476.5,638.83) node    {$f_{3}$};
% Text Node
\draw (430.33,620.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{3,4}$};
% Text Node
\draw (544,639.07) node [anchor=north west][inner sep=0.75pt]    {$t_{1}$};
% Text Node
\draw (548.5,601.83) node    {$f_{1}$};
% Text Node
\draw (551,742.07) node [anchor=north west][inner sep=0.75pt]    {$t_{2}$};
% Text Node
\draw (556.5,705.83) node    {$f_{2}$};
% Text Node
\draw (632,669.07) node [anchor=north west][inner sep=0.75pt]    {$t_{q}$};
% Text Node
\draw (636.5,631.83) node    {$f_{q}$};
% Text Node
\draw (431,773.07) node [anchor=north west][inner sep=0.75pt]    {$t_{5}$};
% Text Node
\draw (436.5,736.83) node    {$f_{5}$};
% Text Node
\draw (589.33,594.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{q,1}$};
% Text Node
\draw (490.33,602.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{1,3}$};
% Text Node
\draw (415.33,702.73) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{3,5}$};
% Text Node
\draw (56.67,1749.07) node [anchor=north west][inner sep=0.75pt]    {$S$};
% Text Node
\draw (183.67,1650.73) node [anchor=north west][inner sep=0.75pt]    {$g$};
% Text Node
\draw (281.33,1732.07) node [anchor=north west][inner sep=0.75pt]    {$f$};
% Text Node
\draw (114.67,1706.07) node [anchor=north west][inner sep=0.75pt]    {$s_{0}$};
% Text Node
\draw (236.67,1800.73) node [anchor=north west][inner sep=0.75pt]    {$t_{0}$};
% Text Node
\draw (34,1649.4) node [anchor=north west][inner sep=0.75pt]  [font=\small]  {$g\circ c_{g,f}( s_{0}) \ =f( t_{0})$};
% Text Node
\draw (241.67,1679.07) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$w_{f,g}$};
% Text Node
\draw (181.33,1826.4) node [anchor=north west][inner sep=0.75pt]    {$T$};
% Text Node
\draw (135.67,1770.73) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$c_{g,f}( s_{0})$};
% Text Node
\draw (134.67,1793.73) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$u_{g,f}( s_{0})$};
% Text Node
\draw (45.17,1794.83) node    {$f$};
% Text Node
\draw (44.33,1831.4) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (114.33,1832.73) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (117.17,1794.83) node    {$g_{i}$};
% Text Node
\draw (71,1772.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{f,g_{i}}$};
% Text Node
\draw (208,325.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$f:\mathtt{t\rightarrow o}$};
% Text Node
\draw (211.33,424.73) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$g:\mathtt{s\rightarrow o} ,\forall g\in G[ s]$};
% Text Node
\draw (69,329) node    {$f$};
% Text Node
\draw (105,501.73) node [anchor=north west][inner sep=0.75pt]    {$s_{n}$};
% Text Node
\draw (65,367.73) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (25,502.4) node [anchor=north west][inner sep=0.75pt]    {$s_{1}$};
% Text Node
\draw (64.33,417.73) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (186.17,331.83) node    {$f$};
% Text Node
\draw (183.33,368.4) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (182.67,418.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (257.67,367) node  [font=\scriptsize]  {$g_{i}$};
% Text Node
\draw (286,331.33) node  [font=\footnotesize]  {$G[ s]$};
% Text Node
\draw (277.67,387) node  [font=\scriptsize]  {$g_{n}$};
% Text Node
\draw (197.5,473.5) node    {$f$};
% Text Node
\draw (196.67,510.07) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (266.67,511.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (269.5,473.5) node    {$g_{i}$};
% Text Node
\draw (223.33,451.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{f,g_{i}}$};
% Text Node
\draw (34,547) node [anchor=north west][inner sep=0.75pt]   [align=left] {Parse Tree};
% Text Node
\draw (182,547) node [anchor=north west][inner sep=0.75pt]   [align=left] {Metaphor (Local)};
% Text Node
\draw (-2,396.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$inconsistent$};
% Text Node
\draw (491,777) node [anchor=north west][inner sep=0.75pt]   [align=left] {Metaphor Graph (Global)};
% Text Node
\draw (21.17,1143) node [anchor=north west][inner sep=0.75pt]  [font=\large] [align=left] {{\fontfamily{pcr}\selectfont Type:}};
% Text Node
\draw (22.33,1168.07) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
\mathtt{boolean}\\
\mathtt{int32}\\
\mathtt{float32}\\
\mathsf{vector}[\mathsf{float} ,\ [ n_{1} ,n_{2} ,...,n_{k}]]\\
\mathsf{Tuple}[ type_{1} ,\ type_{2} ,\ ...]\\
\mathsf{List}[ type,\ n]
\end{array}$};
% Text Node
\draw (17,840.9) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
\mathsf{Learn\ Grounding}\\
\mathsf{Input\ :\ M_{\theta }[ D] ,\ Data[ query,\ grounding]}\\
\mathsf{1.\ init\ the\ model\ M_{\theta }[ D] \ \ where\ D\ is\ the\ domains\ known}\\
\mathsf{\ \ \ } M_{\theta } .vocab\ \mathsf{is\ the\ learned\ vocab,\ each\ word\ associate\ with\ lexicon}\\
while\ not\ done:\\
\ \ \ \mathsf{2.\ filter\ Data\ to\ find\ the\ query\ that\ min\ number\ of\ new\ words\ < \ k}\\
\ \ \ \mathsf{is\ added\ to\ the\ learned\ corpus} .\ \mathsf{NewWords\ is\ collected}\\
\ \ \ \mathsf{3.\ associate\ each\ new\ word\ w\in NewWords\ with\ some\ lexicon\ entries}\\
\ \ \ \mathsf{4.\ grounding\ performed\ on\ the\ newly\ learnd\ words\ and\ lexicon\ entries\ }\\
\mathsf{\ \ \ \ \ \ \ 4.1\ normal\ training}\\
\mathsf{\ \ \ \ \ \ \ 4.2\ infer\ metaphor\ connections}\\
\mathsf{\ \ \ \ \ \ \ 4.3\ repeat\ until\ converge}\\
5.\ \mathtt{return\ } M_{\theta }[ D]
\end{array}$};
% Text Node
\draw (491.43,2154.26) node [anchor=north west][inner sep=0.75pt]    {$D_{0}$};
% Text Node
\draw (471.71,2068.35) node [anchor=north west][inner sep=0.75pt]    {$D_{2}$};
% Text Node
\draw (579.45,2067.64) node [anchor=north west][inner sep=0.75pt]    {$D_{3}$};
% Text Node
\draw (401.99,1983.84) node [anchor=north west][inner sep=0.75pt]    {$D_{5}$};
% Text Node
\draw (507.62,1981.02) node [anchor=north west][inner sep=0.75pt]    {$D_{6}$};
% Text Node
\draw (606.92,1985.25) node [anchor=north west][inner sep=0.75pt]    {$D_{7}$};
% Text Node
\draw (578.75,2153.56) node [anchor=north west][inner sep=0.75pt]    {$D_{1}$};
% Text Node
\draw (671,2097.92) node [anchor=north west][inner sep=0.75pt]    {$D_{4}$};
% Text Node
\draw (529.05,1863.01) node [anchor=north west][inner sep=0.75pt]    {$D_{8}$};
% Text Node
\draw (499.67,1901.34) node [anchor=north west][inner sep=0.75pt]  [font=\small,color={rgb, 255:red, 0; green, 0; blue, 0 }  ,opacity=1 ]  {$w_{f,g}$};
% Text Node
\draw (437.01,1861.9) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (32.31,2153.56) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
Curriculum\ to\ gradually\ learn\ the\ vocab\\
Input\ :\ M_{\theta } ,\ D,\ vocab,\ Data[ corpus,\ grounding]\\
1.\ init\ the\ model\ M_{\theta }[ D] \ \ where\ D\ is\ the\ domains\ known\\
\ \ \ voacb\ is\ initlaized\ with\ the\ intial\ config\\
2.\ filter\ Data[ vocab] \ to\ find\ the\ corpus\ that\ min\ number\ of\ words\ < \ k\\
is\ added\ to\ the\ learned\ corpus\\
3.\ associate\ each\ word\ with\ some\ learnd\ lexicon\ entries\\
4.\ grounding\ performed\ on\ the\ newly\ learnd\ words\ and\ lexicon\ entries\ \\
\ \ \ \ 4.1\ normal\ training\\
\ \ \ \ 4.2\ infer\ metaphor\ connections\\
\ \ \ \ 4.3\ repeat\ until\ converge
\end{array}$};
% Text Node
\draw (546,903.9) node [anchor=north west][inner sep=0.75pt]    {$D_{0}$};
% Text Node
\draw (622,905.9) node [anchor=north west][inner sep=0.75pt]    {$D_{1}$};
% Text Node
\draw (341,1091.9) node [anchor=north west][inner sep=0.75pt]    {$\mathcal{L}( q,x,a) =KL[ \ \mathbb{E}[ parse[ q]( x)] \ ||\ \ a\ ] \ $};
% Text Node
\draw (248,1150.5) node [anchor=north west][inner sep=0.75pt]   [align=left] {1. answer queries on groundings\\2. physics model learned from description of scene\\3. planning using the learned predicates and features/concepts\\4. abstract concepts learned using the unknown};
% Text Node
\draw (134,1912.9) node [anchor=north west][inner sep=0.75pt]    {$dfs\_args( query,\ value)$};
% Text Node
\draw (114,1348) node    {$f$};
% Text Node
\draw (150,1520.73) node [anchor=north west][inner sep=0.75pt]    {$s_{n}$};
% Text Node
\draw (110,1386.73) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (70,1521.4) node [anchor=north west][inner sep=0.75pt]    {$s_{1}$};
% Text Node
\draw (109.33,1436.73) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (79,1566) node [anchor=north west][inner sep=0.75pt]   [align=left] {Parse Tree};
% Text Node
\draw (43,1415.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$inconsistent$};
% Text Node
\draw (257,1332.4) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$f:\mathtt{t\rightarrow o}$};
% Text Node
\draw (260.33,1431.73) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize]  {$g:\mathtt{s\rightarrow o} ,\forall g\in G[ s]$};
% Text Node
\draw (235.17,1338.83) node    {$f$};
% Text Node
\draw (232.33,1375.4) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (231.67,1425.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (306.67,1374) node  [font=\scriptsize]  {$g_{i}$};
% Text Node
\draw (335,1338.33) node  [font=\footnotesize]  {$G[ s]$};
% Text Node
\draw (326.67,1394) node  [font=\scriptsize]  {$g_{n}$};
% Text Node
\draw (246.5,1480.5) node    {$f$};
% Text Node
\draw (245.67,1517.07) node [anchor=north west][inner sep=0.75pt]    {$t$};
% Text Node
\draw (315.67,1518.4) node [anchor=north west][inner sep=0.75pt]    {$s$};
% Text Node
\draw (318.5,1480.5) node    {$g_{i}$};
% Text Node
\draw (272.33,1458.07) node [anchor=north west][inner sep=0.75pt]  [font=\scriptsize,color={rgb, 255:red, 208; green, 2; blue, 27 }  ,opacity=1 ]  {$w_{f,g_{i}}$};
% Text Node
\draw (231,1554) node [anchor=north west][inner sep=0.75pt]   [align=left] {Metaphor (Local)};
% Text Node
\draw (398,1320.4) node [anchor=north west][inner sep=0.75pt]    {$ \begin{array}{l}
\mathsf{Too\ Many\ Functions\ Defined\ on\ type\ S}\\
\\
\mathsf{consider\ the\ input\ type\ as\ objects,\ then\ }\\
\mathsf{almost\ all\ behaviour\ is\ defined\ on\ S\ if\ we}\\
\mathsf{allow\ the\ casting}\\
\\
\mathsf{So\ consider\ the\ functions\ defined\ on\ S}\\
\mathsf{with\ probability\ then\ purge\ the\ p\ percent}
\end{array}$};
% Text Node
\draw (255,1266.4) node [anchor=north west][inner sep=0.75pt]    {$Auto\ Batch\ Functions$};


\end{tikzpicture}
