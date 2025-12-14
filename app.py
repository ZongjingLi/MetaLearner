import tornado.ioloop
import tornado.web
import tornado.httputil
import os
import json
import networkx as nx
from PIL import Image, ImageDraw
import io
import base64
import random
import torch
import numpy as np


data = []

# ------------------------------
# Mock Model Class (with Edge Weights)
# ------------------------------
class Model:
    def __init__(self):
        # Default tree (nx.DiGraph) with edge weights
        self.default_tree = self._create_default_tree()
        # Predefined paths for edges (with weights)
        self.paths = {
            "n1_n2": self._create_path_tree("n1_n2"),
            "n2_n3": self._create_path_tree("n2_n3"),
            "n2_n4": self._create_path_tree("n2_n4")
        }

    def _create_default_tree(self):
        #print("Default:")
        """Create a sample nx.DiGraph with coordinates and edge weights"""
        G = nx.DiGraph()
        # Nodes with (fn, value, type, coordinate)
        nodes = [
            ("n1", {"fn": "load_data", "value": "dataset.csv", "type": "input", "coordinate": (100, 100)}),
            ("n2", {"fn": "preprocess", "value": "clean_data()", "type": "transform", "coordinate": (200, 100)}),
            ("n3", {"fn": "train_model", "value": "lr_model", "type": "train", "coordinate": (300, 80)}),
            ("n4", {"fn": "evaluate", "value": "accuracy=0.92", "type": "eval", "coordinate": (300, 120)})
        ]
        G.add_nodes_from(nodes)

        edges = [
            ("n1", "n2", {"weight": random.uniform(0, 1)}),
            ("n2", "n3", {"weight": random.uniform(0, 1)}),
            ("n2", "n4", {"weight": random.uniform(0, 1)})
        ]
        G.add_edges_from(edges)
        return G

    def _create_path_tree(self, edge_id):
        """Create a path tree for a given edge with weights"""
        G = nx.DiGraph()
        base_nodes = [
            (f"{edge_id}_p1", {"fn": "step1", "value": f"{edge_id}_step1", "type": "path", "coordinate": (50, 50)}),
            (f"{edge_id}_p2", {"fn": "step2", "value": f"{edge_id}_step2", "type": "path", "coordinate": (150, 50)}),
            (f"{edge_id}_p3", {"fn": "step3", "value": f"{edge_id}_step3", "type": "path", "coordinate": (250, 50)})
        ]
        G.add_nodes_from(base_nodes)
        # Path edges with random weights
        edges = [
            (f"{edge_id}_p1", f"{edge_id}_p2", {"weight": random.uniform(0, 1)}),
            (f"{edge_id}_p2", f"{edge_id}_p3", {"weight": random.uniform(0, 1)})
        ]
        G.add_edges_from(edges)
        return G

    def process_query(self, query, files):
        """Process query + files and return tree + paths (with edge weights)"""
        # Convert nx.DiGraph to serializable format (include edge data)
        tree_serialized = {
            "nodes": [
                {"id": n, **self.default_tree.nodes[n]} 
                for n in self.default_tree.nodes
            ],
            "edges": [
                [u, v, self.default_tree.edges[u, v]]  # Include edge data (weight)
                for u, v in self.default_tree.edges
            ]
        }

        return {
            "tree": tree_serialized,
            "paths": []
        }

    def get_edge_paths(self, edge_id):
        """Get path tree for a specific edge (with weights)"""
        path_tree = self.paths.get(edge_id, self._create_path_tree("default"))
        bind = {
            "nodes": [
                {"id": n, **path_tree.nodes[n]} 
                for n in path_tree.nodes
            ],
            "edges": [
                [u, v, path_tree.edges[u, v]]  # Include edge weight
                for u, v in path_tree.edges
            ]
        }
        print("bind", bind["edges"])
        return bind

# ------------------------------
# Mock Visualizer Class (unchanged)
# ------------------------------
class Visualizer:
    def __init__(self):
        pass

    def update_nodes_info(tree, paths):
        return


    def visualize_node(self, node_id):
        """Generate node visualization (image + info)"""
        # Create a simple PIL image for demo
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 40), f"Node: {node_id}", fill='black')
        draw.rectangle([10, 10, 190, 90], outline='#00629B', width=2)

        # Convert image to base64 URL
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        img_url = f"data:image/png;base64,{img_base64}"

        node_info = {
            "id": node_id,
            "fn": f"function_{node_id}",
            "value": f"value_{node_id}",
            "type": f"type_{node_id[-1]}",
            "coordinate": (100 + int(node_id[-1])*50, 100 + int(node_id[-1])*30),
            "metadata": "Sample metadata for demo"
        }

        return {
            "image_url": img_url,
            "node_info": node_info
        }

# ------------------------------
# Tornado Handlers (with Default Tree Weights)
# ------------------------------
class BaseHandler(tornado.web.RequestHandler):
    """Base handler with JSON error handling"""
    def write_error(self, status_code, **kwargs):
        self.set_header("Content-Type", "application/json")
        self.write(json.dumps({
            "error": {
                "code": status_code,
                "message": self._reason
            }
        }))

class MainHandler(BaseHandler):
    def get(self):
        self.render("assets/index.html")

class DefaultTreeHandler(BaseHandler):
    def initialize(self, model):
        self.model = model

    def get(self):
        try:
            # Return default tree with edge weights
            assert 0
            tree = {
                "nodes": [
                    {"id": n, **self.model.default_tree.nodes[n]} 
                    for n in self.model.default_tree.nodes
                ],
                "edges": [
                    [u, v, self.model.default_tree.edges[u, v]]
                    for u, v in self.model.default_tree.edges
                ]
            }
            self.write(tree)
            #print("Default Tree:",tree)
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error ons": str(e)}))

class ModelHandler(BaseHandler):
    def initialize(self, model):
        self.model = model
 

    def post(self):
        try:
            # Parse form data
            query = self.get_argument('query', '')

            files = self.request.files.get('files', [])
            #tags = self.request.files.get('tags', [])
            #tags = self.get_argument("tags", [])
            tags = self.get_arguments('file_tags')

            grounding = {}

 
            for i,tag in enumerate(tags):
                grounding[tag] = files[i]["filename"]
                if files[i]["content_type"][:5] == "image":
                    img = Image.open(io.BytesIO(files[i]['body'])).convert('RGB')
                    grounding[tag] = torch.tensor(np.array(img)).permute(2,0,1) / 255.0 



            result = self.model.process_query(query, grounding)
            data.append(result)

            self.write(result)
            
        except Exception as e:
            import traceback
            print(f"[ERROR] /api/model failed: {str(e)}")
            print(traceback.format_exc())
            self.set_status(500)
            self.write(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
        


class EdgePathsHandler(BaseHandler):
    def initialize(self, model):
        self.model = model

    def get(self):
        try:
            edge_id = self.get_argument('edge_id', '')
            path_tree = self.model.get_edge_paths(edge_id)
            self.write({"path_tree": path_tree})
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

class VisualizeNodeHandler(BaseHandler):
    def initialize(self, model):
        self.model = model

    def get(self):
        try:
            print(self.model.result)
            node_id = self.get_argument('node_id', '')
            visualization = self.visualizer.visualize_node(node_id)
            self.write(visualization)
        except Exception as e:
            self.set_status(500)
            self.write(json.dumps({"error": str(e)}))

# ------------------------------
# App Setup
# ------------------------------
def make_app(model, visualizer):
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/api/default-tree", DefaultTreeHandler, {"model": model}),
        (r"/api/model", ModelHandler, {"model": model}),
        (r"/api/edge-paths", EdgePathsHandler, {"model": model}),
        (r"/api/visualize-node", VisualizeNodeHandler, {"model": model}),
    ],
    template_path=os.path.dirname(__file__),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    debug=True)

if __name__ == "__main__":

    from core.model import MetaLearner
    learner = MetaLearner([])
    learner.load_ckpt("outputs/checkpoints/max1")

    from domains.visualizer import DomainVisualizer
    visualizer = DomainVisualizer()
    visualizer = Visualizer() 

    learner.visualizer = visualizer

    #learner = Model()

    app = make_app(learner)
    app.listen(8888)
    print("Tornado app running at http://localhost:8888")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        tornado.ioloop.IOLoop.current().stop()