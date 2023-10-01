import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn.primary import *
from .executor import *
from .programs.symbolic import *
from Karanir.utils import *

class LanguageModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.vectorize = VectorConstructor(config)
        self.vector_encoder = nn.GRU(config.word_dim,int(config.semantics_dim/2), batch_first =True, bidirectional=True) 
        self.arg_embeddings = nn.Embedding(int(1e1), config.semantics_dim)
        self.continuer = FCBlock(100,3,config.semantics_dim*2, config.semantics_dim)
        self.semantics2token =  FCBlock(100,3,config.semantics_dim*2, config.token_dim)
        self.EPS = 1e-6


    def forward(self,x):
        repres = [self.vectorize.get_word_vectors(xs) for xs in x]
        seqes  = [self.vector_encoder(wvs)[0] for wvs in repres]

        return {"seq_features":seqes}

    def translate(self, inputs, lib, executor, targets = None):
        """
        lib: [**{tokens,features,args}]
        """
        repres = [self.vectorize.get_word_vectors(xs) for xs in inputs]
        seqes_features = [self.vector_encoder(wvs)[0][-1:,:] for wvs in repres]

        if targets is not None:
            target_qs = [executor.parse(target) for target in targets]
        else:target_qs = [None for _ in seqes_features]
        
        parse_results = []
        losses = []
        programs = []
        for i,feature in enumerate(seqes_features):
            curr_outputs = self.decode(feature, lib[i], target_qs[i])
            losses.append(curr_outputs["loss"])
            programs.append(curr_outputs["program"])
        outputs = {"loss":losses,"program":programs}
        return outputs

    def decode(self,feature, lib, target_q):
        def parse_node(prior_feature,control_feature,target = None):
            token_names = lib["tokens"]
            token_feature = lib["features"]
            joint_continue = self.continuer(torch.cat([prior_feature, control_feature], dim = -1))
            joint_token_feature = self.semantics2token(torch.cat([prior_feature, control_feature], dim = -1))

            #token_distribution = torch.sigmoid(torch.einsum("nd,md->nm",).squeeze(0))
            token_distribution = torch.matmul(joint_token_feature, token_feature.permute(1,0)).squeeze(0)
            token_distribution = torch.softmax(token_distribution, dim = -1)
            pdf = np.array(token_distribution.detach())

            if target is not None:
                if isinstance(target,str):target_name = target
                else: target_name = target.__class__.__name__

                arg_num = lib["args_num"][token_names.index(target_name)]
                target_index = token_names.index(target_name)
            else:
                target_index = np.random.choice(list(range(len(token_names))),p = pdf/np.sum(pdf))
                target_name = token_names[target_index]
                arg_num = lib["args_num"][token_names.index(target_name)]
                
            
            if arg_num != 0:
                parsed_node = target_name + "(" +"{}," * (arg_num-1) + "{}" +")"
            else: parsed_node = target_name

            loss = 0 - torch.log(token_distribution[target_index] + self.EPS)

            child_node = None; right_child = None; left_child = None
            
            if arg_num == 1:
                if target is not None:child_node = target.child
                arg_feature = self.arg_embeddings(torch.tensor([0]))
                #print(arg_feature.shape)
                outputs = parse_node(joint_continue,arg_feature[0:1],child_node)

                # [After Parse]
                loss += outputs["loss"]
                parsed_node = parsed_node.format(outputs["program"])

            if arg_num == 2:
                if target is not None:left_child = target.left_child
                if target is not None:right_child = target.right_child
                arg_feature = self.arg_embeddings(torch.tensor([0,1]))
         
                left_outputs = parse_node(joint_continue,arg_feature[0:1], left_child)
                right_outputs = parse_node(joint_continue,arg_feature[1:2], right_child)

                # [After Parse]
                loss += left_outputs["loss"] + right_outputs["loss"]
                parsed_node = parsed_node.format(left_outputs["program"],right_outputs["program"])


            return {"loss":loss,"program":parsed_node}
        start_feature = torch.zeros_like(feature)

        outputs = parse_node(feature,start_feature,target_q)

        return outputs

class VectorConstructor(nn.Module):
    def __init__(self,config):
        super().__init__()
        # construct vector embeddings and other utils for given corpus
        corpus_path = config.corpus_path

        with open(corpus_path) as corpus_loaded:
            corpus = [t.strip() for t in corpus_loaded]
            self.corpus = corpus
            self.key_words = []
            self.token_to_id = build_vocab(corpus)
        self.id_to_token = reverse_diction(self.token_to_id)

        self.word_vectors = nn.Embedding(config.num_words,config.word_dim)

    def forward(self,sentence):
        return sentence
    
    def get_word_vectors(self,sentence):
        code = encode(tokenize(sentence),self.token_to_id)
        word_vectors = self.word_vectors(torch.tensor(code))
        return word_vectors

