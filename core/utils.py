
from typing import List

def load_corpus(corpus_name : str) -> List[str]:    
    sequences = []
    with open() as corpus:
        for line in corpus:
            line = line.strip()
            if line:
                line = line.lower()
                line = ' '.join(line.split())
                sequences.append(line)
    return sequences