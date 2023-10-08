from visualization import *
from .models import *
from config import *

visparser = argparse.ArgumentParser()
visparser.add_argument("--checkpoint",                 default = "checkpoints/alueth.pth")
visparser.add_argument("--logflag",                    default = False)

visconfig = visparser.parse_args()

model = MetaLearner(config)
model.load_state_dict(torch.load(visconfig.checkpoint))

concepts = ["topological_vector_space",
            "banach_steinhaus_theorem",
            "close", "open", "compact",
            "seperation", "banach_space",
            "hilbert_space", "heine_borel",
            "bounded", "continuous", "function",
            "map", "mapping","manifold","complete","normed","has_inner_product"]


visualize_concepts(concepts,model, visconfig.logflag)
plt.show()