from models import *
from Aluneth.config import *

langs = LanguageModel(config)
metar = MetaReasoner(config)

output_seqs = langs([
    "what is that ?",
    "what is Banach-Steinhaus theorem ?",
    "what is that concept ?"])

print(metar.executor.parse("filter(scene(),banach_space)"))
print(metar.executor.get_concept_embedding("banach_space").shape)

for s in output_seqs["seq_features"]:
    print(s.shape)

# this is the Aluneth