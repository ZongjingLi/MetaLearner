
import os
import json
import tornado.ioloop
import tornado.web
import tornado.escape
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import torch
from core.metaphors.diagram import ConceptDiagram, MetaphorMorphism

import json

device = "mps" if torch.backends.mps.is_available() else "cpu"

def save_concept_diagram_to_json(concept_diagram, file_path="concept_diagram.json"):
    """
    Saves the ConceptDiagram object into a JSON format.
    """
    data = {
        "nodes": [],
        "edges": []
    }
    
    # Extract domains as nodes
    for domain_name, domain in concept_diagram.domains.items():
        data["nodes"].append({
            "data": {"id": domain_name, "label": domain_name}
        })
    
    # Extract morphisms as edges
    for (source, target), morphisms in concept_diagram.edge_indices.items():
        for morphism_name in morphisms:
            data["edges"].append({
                "data": {
                    "id": morphism_name.split("_")[-1],
                    "source": source,
                    "target": target,
                    "label": morphism_name.split("_")[-1]
                }
            })
    
    # Save to file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Concept diagram saved to {file_path}")

def get_concept_diagram_json(concept_diagram):
    """
    Convert the ConceptDiagram object to a JSON format suitable for visualization.
    """
    data = {
        "nodes": [],
        "edges": []
    }
    
    # Extract domains as nodes
    for domain_name, domain in concept_diagram.domains.items():
        data["nodes"].append({
            "data": {
                "id": domain_name.strip(),
                "label": domain_name.strip(),
                "type": "domain"
            }
        })
    
    # Extract morphisms as edges
    for (source, target), morphisms in concept_diagram.edge_indices.items():
        for morphism_name in morphisms:
            label = morphism_name.split("_")[-1]  # Extract last number after _
            data["edges"].append({
                "data": {
                    "id": morphism_name.strip(),
                    "source": source.strip(),
                    "target": target.strip(),
                    "label": label,
                    "type": "morphism"
                }
            })
    
    return data

# Initialize the ConceptDiagram object
concept_diagram = ConceptDiagram().to(device)
# Example: Populate concept_diagram with domains and morphisms (adjust based on your provided structure)
# concept_diagram.add_domain("GenericDomain", generic_executor)
# concept_diagram.add_morphism("GenericDomain", "LineDomain", MetaphorMorphism(...))

if True:
    from core.metaphors.diagram import ConceptDiagram, MetaphorMorphism
    from domains.generic.generic_domain import generic_executor
    from domains.line.line_domain import line_executor
    from domains.rcc8.rcc8_domain import rcc8_executor
    from domains.curve.curve_domain import curve_executor
    from domains.distance.distance_domain import distance_executor
    from domains.direction.direction_domain import direction_executor
    from domains.pointcloud.pointcloud_domain import pointcloud_executor

    concept_diagram = ConceptDiagram()
    curve_executor.to(device)

    domains = {
    "GenericDomain": generic_executor,
    "LineDomain": line_executor,
    "CurveDomain": curve_executor,
    "RCC8Domain": rcc8_executor,
    "DistanceDomain": distance_executor,
    "DirectionDomain": direction_executor,
    "PointcloudDomain": pointcloud_executor
    }

    for domain_name, executor in domains.items(): concept_diagram.add_domain(domain_name, executor)

    morphisms = [
    ("GenericDomain", "LineDomain"),
    ("GenericDomain", "DistanceDomain"),
    ("GenericDomain", "DirectionDomain"),
    ("DistanceDomain", "DirectionDomain"),
    ("CurveDomain", "LineDomain"),
    ("LineDomain", "RCC8Domain"),
    ("LineDomain", "RCC8Domain"),
    ("DistanceDomain", "RCC8Domain"),
    ("GenericDomain", "CurveDomain"),
    ("GenericDomain", "PointcloudDomain")
    ]

    for source, target in morphisms:
        concept_diagram.add_morphism(source, target, MetaphorMorphism(domains[source], domains[target]))


    save_concept_diagram_to_json(concept_diagram, "assets/diagram-json")

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
class DiagramHandler(tornado.web.RequestHandler):
    def initialize(self, concept_diagram):
        self.concept_diagram = concept_diagram

    def get(self):
        diagram_json = get_concept_diagram_json(self.concept_diagram)
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps(diagram_json))

class DomainSummaryHandler(tornado.web.RequestHandler):
    def initialize(self, concept_diagram):
        self.concept_diagram = concept_diagram

    def get(self, domain_id):
        domain_id = domain_id.strip()
        if domain_id in self.concept_diagram.domains:
            domain = self.concept_diagram.domains[domain_id]
            summary = domain.domain.get_summary()
            structured_summary = f"""
Domain: {domain_id}
-----------------------
{summary}
"""
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"summary": structured_summary}, indent=4))
        else:
            print(f"Domain not found: {domain_id}. Available: {list(self.concept_diagram.domains.keys())}")
            self.set_status(404)
            self.write(json.dumps({"error": "Domain not found"}, indent=4))


class MetaphorDetailsHandler(tornado.web.RequestHandler):
    def initialize(self, concept_diagram):
        self.concept_diagram = concept_diagram

    def get(self, morphism_id):
        morphism_id = morphism_id.strip()
        for (source, target), morphisms in self.concept_diagram.edge_indices.items():
            if morphism_id in morphisms:
                morphism = self.concept_diagram.morphisms[morphism_id]
                structured_details = f"""
Metaphor Mapping
-----------------------
Source Domain: {source.strip()}
Target Domain: {target.strip()}
Morphism Name: {morphism_id}
"""
                self.set_header("Content-Type", "application/json")
                self.write(json.dumps({"details": structured_details}, indent=1))
                return
        
        print(f"Metaphor not found: {morphism_id}. Available: {self.concept_diagram.edge_indices}")
        self.set_status(404)
        self.write(json.dumps({"error": "Metaphor not found"}, indent=4))
    
class ExecuteCodeHandler(tornado.web.RequestHandler):
    def initialize(self, concept_diagram, state_store):
        self.concept_diagram = concept_diagram
        self.state_store = state_store  # Shared storage for state visualization

    def post(self):

            predicate = self.get_argument("code").strip()
            if not predicate:
                self.set_status(400)
                self.write(json.dumps({"error": "Predicate cannot be empty"}))
                return

            print(f"Executing predicate: {predicate}")

            # Generate random input state
            source_state = torch.randn([3, 256], requires_grad=False).to(device)
            context = {0: {"state": source_state}, 1: {"state": source_state}}

            domain_name = "GenericDomain"
            if domain_name not in self.concept_diagram.domains:
                raise ValueError(f"Domain '{domain_name}' not found in concept diagram")

            # Evaluate predicate
            self.concept_diagram.to(device)
 
            evaluation_result = self.concept_diagram.evaluate(source_state, predicate, domain_name, eval_type="metaphor")

            # Extract path details
            import random
            idx = random.randint(0, len(evaluation_result["apply_path"]) - 1)
    
            apply_path = evaluation_result["apply_path"][idx]
            state_path = evaluation_result["state_path"][idx]
            metas_path = evaluation_result["metas_path"][idx]

            # Store states for visualization
            path_data = []
            path_edges = []
            symb_edges = []


            visualizations = self.concept_diagram.visualize_path(state_path, metas_path, evaluation_result["results"][idx].cpu().detach())
            
            for i, ((src_domain, tgt_domain, morphism_index), state) in enumerate(zip(metas_path, state_path[1:])):
                state_id = f"path_state_{i}"
                self.state_store[state_id] = state.detach().cpu().numpy()
                edge_id = f"morphism_{src_domain}_{tgt_domain}_{morphism_index}"

                path_data.append({
                    "id": state_id,
                    "source": src_domain,
                    "target": tgt_domain,
                    "morphism": edge_id,
                    "apply_prob": float(apply_path[i])
                })
                path_edges.append(edge_id)
            

            # Store initial source and final target states separately
            self.state_store["source_domain"] = source_state.detach().cpu().numpy()
            self.state_store["target_domain"] = state_path[-1].detach().cpu().numpy()

            # Store connection matrix for visualization
            last_morphism = metas_path[-1]
            last_morphism_key = f"morphism_{last_morphism[0]}_{last_morphism[1]}_{last_morphism[2]}"
            if last_morphism_key in self.concept_diagram.morphisms:
                connection_matrix = self.concept_diagram.morphisms[last_morphism_key].predicate_matrix()[0]
                self.state_store["connection_matrix"] = connection_matrix.detach().cpu().numpy()

            # Package visualizations as base64 for inline display
            visualized_steps = []
            for vis in visualizations:
                visualized_steps.append({
                    "step": vis["step"],
                    "source": vis["source"],
                    "target": vis["target"],
                    "image": vis["image"]  # base64-encoded image
                })

            # Response data
            symbs = [float(item) if isinstance(item, torch.Tensor) else item for item in evaluation_result["symbol_path"][idx]]
            response_data = {
                "result": evaluation_result["results"][idx].detach().cpu().tolist(),
                "symbs" : symbs,
                "path": path_data,
                "path_edges": path_edges,
                "visualizations": visualized_steps
            }

            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(response_data))


            #print(f"Error executing predicate: {e}")
            #self.set_status(500)
            #self.write(json.dumps({"error": str(e)}))

class StateVisualizationHandler(tornado.web.RequestHandler):
    def initialize(self, concept_diagram, state_store):
        self.concept_diagram = concept_diagram
        self.state_store = state_store  # Shared state store

    def get(self, state_id):
        try:
            # Retrieve stored states

            if state_id in self.state_store:
                state = self.state_store[state_id]
            else:
                self.set_status(404)
                self.write({"error": "State not found"})
                return

            # Convert state to image
            fig, ax = plt.subplots()
            ax.imshow(state, cmap="coolwarm")
            ax.axis("off")
            import io

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode("utf-8")

            self.set_header("Content-Type", "application/json")
            self.write({"image": img_base64})

        except Exception as e:
            self.set_status(500)
            self.write({"error": str(e)})

if __name__ == "__main__":
    settings = {
        "template_path": os.path.join(os.path.dirname(__file__), "templates"),
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
        "debug": True,
    }

    state_store = {}

    app = tornado.web.Application([
    (r"/", MainHandler),
    (r"/diagram", DiagramHandler, {"concept_diagram": concept_diagram}),
    (r"/domain-summary/(.+)", DomainSummaryHandler, dict(concept_diagram=concept_diagram)),
    (r"/morphism-details/(.+)", MetaphorDetailsHandler, dict(concept_diagram=concept_diagram)),
    (r"/execute", ExecuteCodeHandler, {"concept_diagram": concept_diagram, "state_store": state_store}),
    (r"/visualize-state/(.+)", StateVisualizationHandler, {"concept_diagram": concept_diagram, "state_store": state_store})
    ], **settings)


    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()
