import tornado.ioloop
import tornado.web
import tornado.websocket
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

class MainHandler(tornado.web.RequestHandler):
    def initialize(self, model=None):
        self.model = model

    def get(self):
        self.render("index.html")

class UploadHandler(tornado.web.RequestHandler):
    def initialize(self, model=None):
        self.model = model
    def post(self):
        try:
            if 'file' not in self.request.files:
                raise ValueError("No file provided")
                
            file_info = self.request.files['file'][0]
            original_filename = file_info['filename']
            
            # Create a safe filename
            filename = ''.join(c for c in original_filename if c.isalnum() or c in '._-')
            
            # Ensure assets/uploads directory exists
            upload_dir = os.path.join("assets", "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save uploaded file
            output_file = os.path.join(upload_dir, filename)
            with open(output_file, 'wb') as out:
                out.write(file_info['body'])
            
            # Generate URL path for the uploaded file
            file_url = f"/assets/uploads/{filename}"
            
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps({
                "status": "success",
                "filename": filename,
                "url": file_url
            }))
        except Exception as e:
            self.set_status(400)
            self.write(json.dumps({
                "status": "error",
                "message": str(e)
            }))

class WebSocketHandler(tornado.websocket.WebSocketHandler):
    connections = set()

    def open(self):
        WebSocketHandler.connections.add(self)

    def on_close(self):
        WebSocketHandler.connections.remove(self)

    def on_message(self, message):
        data = json.loads(message)
        if data['type'] == 'code':
            try:
                exec(data['code'])
                self.write_message({"type": "code_result", "result": "Code executed successfully"})
            except Exception as e:
                self.write_message({"type": "code_result", "result": str(e)})
        elif data['type'] == 'update_dag':
            self.update_dag(data['dag_data'])

    def update_dag(self, dag_data):
        G = nx.DiGraph(dag_data)
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, arrowsize=20)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        self.write_message({
            "type": "dag_update",
            "image": image_base64
        })

def make_app(model = None):
    base_dir = os.path.dirname(__file__)
    assets_dir = os.path.join(base_dir, "assets")
    uploads_dir = os.path.join(assets_dir, "uploads")
    
    # Ensure assets and uploads directories exist
    os.makedirs(uploads_dir, exist_ok=True)
    
    return tornado.web.Application([
        (r"/", MainHandler, {"model" : model}),
        (r"/upload", UploadHandler, {"model" : model}),
        (r"/websocket", WebSocketHandler),
        (r"/assets/uploads/(.*)", tornado.web.StaticFileHandler, {"path": uploads_dir}),
    ], 
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    print("Server running on http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()