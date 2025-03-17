
from rinarak.domain import load_domain_string, Domain

from rinarak.dsl.logic_primitives import *

demo_domain_string = f"""
(domain scourge_demo)
(:type
    object - vector[float,100]
    position - vector[float,2]
    color - vector[float, 64]
    category
)
(:predicate
    color ?x-object -> vector[float,64]
    is-red ?x-object -> boolean
    is-blue ?x-object -> boolean
    is-ship ?x-object -> boolean
    is-house ?x-object -> boolean
    left ?x-object ?y-object -> boolean
)
(:derived
    is-green ?x-color expr: (??f ?x)
)
(:constraint
    (color: is-red is-blue)
    (category: is-ship is-house)
)
(:action
    (
        name:pick-up
        parameters:?x-object ?y
        precondition: (is-red (color ?x))
        effect: (
            if (is-blue ?y)
                (and
                (assign (is-blue ?x) 1.0 )
                (assign (is-red ?y) 0.0)
                )
            )
    )
)
"""

icc_parser = Domain("rinarak/base.grammar")
domain = load_domain_string(demo_domain_string, icc_parser)
domain.print_summary()

from rinarak.knowledge.executor import CentralExecutor
from rinarak.dsl.logic_primitives import *
executor = CentralExecutor(domain)
executor.redefine_predicate("is-red", 
                            lambda x: {**x, "end":torch.min(x["end"], torch.tensor([-4.0,-2.0]))}
                            )


outputs = executor.evaluate("( (is-red $0) )", {0:
                                  {"end": torch.tensor([-1.,1.]),"features": torch.randn([2,100]), "executor": executor}
                                  })

outputs = executor.evaluate("( relate $0 $0 left )", {0:
                                  {"end": torch.tensor([-1.,1.]),"features": torch.randn([2,100]), "relations": torch.randn([2,2,100]), "executor": executor}
                                  })

print(outputs["end"])

for derived in domain.derived:print(derived, domain.derived[derived])

def fit_scenes(dataset, model, epochs = 1, batch_size = 2, lr = 2e-4, verbose = True):
    import sys
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size)
    # use Adam optimizer for the optimization of embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for sample in loader:
            """Load the scene features and other groundings"""
            features = sample["scene"]["features"]
            percept_end, features = model.perception(features)

            if sample["scene"]["end"] is not None:
                end = sample["scene"]["end"]
                end = logit(end)
                end = torch.min(end, percept_end)
            else:
                end = percept_end

            """Loss calculation for the grounding modules"""
            loss = 0.0
            programs = sample["programs"]
            answers = sample["answers"]

            # perform grounding on each set of scenes
            batch_num = features.shape[0]
            acc = 0; total = 0
            for b in range(batch_num):
                for i,program_batch in enumerate(programs):
                    program = program_batch[b]
                    answer = answers[i][b]
                    """Context for the central executor on the scene"""
                    context = {
                        "end":end[b],
                        "features":features[b],
                        "executor":model.central_executor
                    }
                    q = Program.parse(program)
                    output = q.evaluate({0:context})
                    
                    if answer in ["yes","no"]:
                        if answer == "yes":loss -= torch.log(output["end"].sigmoid())
                        else:loss -= torch.log(1 - output["end"].sigmoid())
                        if output["end"].sigmoid() > 0.5 and answer == "yes": acc += 1
                        if output["end"].sigmoid() < 0.5 and answer == "no": acc += 1
                    else:
                        loss += torch.abs(output["end"] - int(answer))
                        if torch.abs(output["end"] - int(answer)) < 0.5: acc += 1
                    total += 1
                loss /= len(programs) # program wise normalization
            loss /= batch_num # batch wise normalization
            """Clear gradients and perform optimization"""
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().numpy()
            if verbose:
                sys.stdout.write(f"\repoch:{epoch+1} loss:{str(loss)[7:12]} acc:{acc/total}[{acc}/{total}]")
    if verbose:sys.stdout.write("\n")