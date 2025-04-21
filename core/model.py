import torch
import torch.nn as nn
from typing import List, Union, Any
from helchriss.domain import load_domain_string
from helchriss.knowledge.symbolic import Expression
from helchriss.knowledge.executor import CentralExecutor
from core.metaphors.diagram_executor import MetaphorExecutor
from core.grammar.ccg_parser import ChartParser
from core.grammar.lexicon import CCGSyntacticType, LexiconEntry, SemProgram
from core.grammar.learn import enumerate_search
from helchriss.knowledge.symbolic import Expression
from helchriss.dsl.dsl_values import Value

from tqdm import tqdm

from torch.utils.data import DataLoader
from helchriss.utils.data import ListDataset
from helchriss.utils.data import GroundBaseDataset

class SceneGroundingDataset(ListDataset):
    def __init__(self, queries : List[str], answers : List[Union[Value, Any]], groundings : None):
        query_size = len(queries)
        if groundings is None: groundings = [{} for _ in range(query_size)]
        data = [{"query":queries[i], "answer":answers[i], "grounding": groundings[i]} for i in range(query_size)]
        super().__init__(data)


class Aluneth(nn.Module):
    def __init__(self, domains : List[Union[CentralExecutor]], vocab = None):
        super().__init__()
        self._domain :List[Union[CentralExecutor]]  = domains
        self.executor : CentralExecutor = MetaphorExecutor(domains)
        
        self.vocab = vocab
        self.parser = None
        self.gather_format = self.executor.gather_format

        self.entries_setup()
    
    def freeze_word(self,word):
        for entry in self.parser.word_weights:
            if word in entry:
                self.parser.word_weights[entry]._requires_grad = False
    
    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        metadata = checkpoint['metadata']

        domains = metadata["domains"]

        self.load_state_dict(checkpoint['model_state_dict'])
        return 0
    
    def save_ckpt(self, ckpt_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'metadata': {
                'domains': [],
            }
        }, ckpt_path)
        return 0

    @property
    def domains(self):
        gather_domains = []
        for domain in self._domain:
            if isinstance(domain, CentralExecutor):
                gather_domains.append(domain.domain)
            else: gather_domains.append(domain)
        return gather_domains

    def collect_funcs(self,func_bind, domain):
        bind = {
            "name" : func_bind["name"],
            "parameters" : [self.gather_format(param.split("-")[-1], domain)  for param in func_bind["parameters"]],
            "type" : self.gather_format(func_bind["type"], domain)
            }
        return bind
    
    @property
    def types(self):
        domain_types = {}
        for domain in self.domains:
            for tp in domain.types:
                domain_types[self.gather_format(tp, domain.domain_name)] =  domain.types[tp].replace("'","")
        return domain_types

    @property
    def functions(self):
        domain_functions = {}
        for domain in self.domains:
            domain_name = domain.domain_name
            for func in domain.functions:
                domain_functions[self.gather_format(func, domain_name)] =  self.collect_funcs(domain.functions[func], domain_name)
        return domain_functions

    def entries_setup(self, depth = 1):
        self.entries = enumerate_search(self.types, self.functions, max_depth = depth)
        #for syn_type, program in self.entries:
        #    print(syn_type, program)
        lexicon_entries = {} 
        for word in self.vocab:
            lexicon_entries[word] = []
            for syn_type, program in self.entries:

                lexicon_entries[word].append(LexiconEntry(
                    word, syn_type, program, weight = torch.tensor(-0.0, requires_grad=True)
                ))

        self.lexicon_entries = lexicon_entries
        self.parser = ChartParser(lexicon_entries)

    def forward(self, sentence, grounding = None, topK = None):
        parses = self.parser.parse(sentence, topK = topK)
        log_distrs = self.parser.get_parse_probability(parses)
        
        results = []
        probs = []
        programs = []
        for i,parse in enumerate(parses):
            parse_prob = log_distrs[i]
            program = parse.sem_program
            output_type = self.functions[program.func_name]["type"]

            if len(program.lambda_vars) == 0:
                expr = Expression.parse_program_string(str(program))
                result = self.executor.evaluate(expr, grounding)

                results.append(Value(output_type.split(":")[0],result))
                probs.append(parse_prob)
                programs.append(str(program))
            else:
                results.append(None)
                probs.append(parse_prob)

        return results, probs, programs

    def train(self, dataset : SceneGroundingDataset,  epochs : int = 100, lr = 1e-2, topK = None):
        import tqdm.gui as tqdmgui
        optim = torch.optim.Adam(self.parameters(), lr = lr)
        # epoch_bar = tqdmgui.tqdm(range(epochs), desc="Training epochs", unit="epoch")

        epoch_bar = tqdm(range(epochs), desc="Training epochs", unit="epoch")

        for epoch in epoch_bar:
            loss = 0.0     
            for idx, sample in dataset:

                query = sample["query"]
                answer = sample["answer"]
                grounding = sample["grounding"]

                results, probs, programs = self(query, grounding, topK)
                if not results: print(f"no parsing found for query:{query}")
                for i,result in enumerate(results):
                    measure_conf = torch.exp(probs[i])
                    if result is not None: # filter make sense progams
                        assert isinstance(result, Value), f"{programs[i]} result is :{result} and not a Value type"

                        if answer.vtype == result.vtype:
                            if answer.vtype == "boolean":
                                measure_loss =  torch.nn.functional.binary_cross_entropy_with_logits(result.value - answer.value)
                            if answer.vtype == "int" or answer.type == "float":
                                measure_loss = torch.abs(result.value - answer.value)
                            loss += measure_conf * measure_loss
                        else: loss += measure_conf # suppress type-not-match outputs
                    else: loss += measure_conf # suppress the non-sense outputs
            optim.zero_grad()
            loss.backward()
            optim.step()

            avg_loss = loss / len(dataset) if len(dataset) > 0 else 0
            #epoch_bar.set_postfix({"avg_loss": f"{avg_loss.item():.4f}"})

        return self

    def parse_display(self, sentence):
        parses = self.parser.parse(sentence)
        distrs = self.parser.get_parse_probability(parses)
        parse_with_prob = list(zip(parses, distrs))
        sorted_parses = sorted(parse_with_prob, key=lambda x: x[1], reverse=True)
        for i, parse in enumerate(sorted_parses[:4]):
            print(f"{parse[0].sem_program}, {parse[1].exp():.2f}")
        print("")

    def detect_metaphors(self, sentence : Union[List[str], str]):
        """for the input corpus, use the current model to try to find the appropriate type casting
        this method tries to parse the sentences using the current learned entries. If cannot merge, then add the type casting
        Args:
            sentences: could be a single sentence or a list of sentence
        """
        assert isinstance(self.executor, MetaphorExecutor), "current executor does not support the type mapping"
        #self.executor.add_caster()
        return 