import os
import tornado.ioloop
import tornado.web
import tornado.websocket
import json

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
        

class ProcessHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model
        
    def post(self):
        input_text = self.get_argument("input_text", "")
        
        # Handle file uploads if any
        file_info = None

        if self.request.files and 'file' in self.request.files:
            uploaded_file = self.request.files['file'][0]
            filename = uploaded_file['filename']
            content_type = uploaded_file['content_type']
            file_body = uploaded_file['body']
            
            # Save the file
            save_path = os.path.join(os.path.dirname(__file__), "static/uploads")
            os.makedirs(save_path, exist_ok=True)
            file_path = os.path.join(save_path, filename)
            
            with open(file_path, 'wb') as f:
                f.write(file_body)
                
            file_info = {
                "filename": filename,
                "content_type": content_type,
                "path": f"/static/uploads/{filename}"
            }
        
        # Process both text and file using the model
        grounding = None
        if file_info:
            # Handle image processing for grounding
            if file_info['content_type'].startswith('image/'):
                # This would be replaced with actual image processing code
                from torchvision import transforms
                from PIL import Image

                try:
                    img_path = os.path.join(os.path.dirname(__file__), "static/uploads", file_info['filename'])
                    img = Image.open(img_path)
                    transform = transforms.ToTensor()
                    tensor_img = transform(img)
                    grounding = {"input_image": tensor_img}
                except Exception as e:
                    print(f"Error processing image: {e}")
                    grounding = {}

        # Process the input with the model
        values, weights, programs = self.model.verbose_call(input_text, grounding)
        
        # Format the results
        items = []
        for i in range(len(values)):
            items.append(f"{programs[i]} -> {values[i].value}-{values[i].vtype} P:{weights[i]:.2f} ")

        # Generate graph data after processing
        graph_data = json.loads(json.dumps(self.model.eval_graph()))

        # Return results as JSON
        self.write(json.dumps({
            "list_items": items,
            "graph_data": graph_data,
            "file_info": file_info
        }))

class GraphDataHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model
        
    def get(self):
        """
        Return the graph data for visualization
        """
        # Get graph data from model
        graph_data = self.model.display()
        
        # Set JSON header
        self.set_header("Content-Type", "application/json")
        self.write(graph_data)

def make_app(model=None):
    static_path = os.path.join(os.path.dirname(__file__), "static")
    template_path = os.path.join(os.path.dirname(__file__), "templates")
    
    # Create static directory if it doesn't exist
    os.makedirs(static_path, exist_ok=True)
    os.makedirs(template_path, exist_ok=True)
    
    # Create static/images directory for the parse tree images
    images_path = os.path.join(static_path, "images")
    os.makedirs(images_path, exist_ok=True)
    
    # Create uploads directory for user files
    uploads_path = os.path.join(static_path, "uploads")
    os.makedirs(uploads_path, exist_ok=True)
    
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/process", ProcessHandler, dict(model=model)),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": static_path}),
    ], 
    template_path=template_path,
    static_path=static_path,
    debug=True)

if __name__ == "__main__":
    app = make_app(model="demo_model")
    app.listen(8888)
    print("Server started at http://localhost:8888")
    tornado.ioloop.IOLoop.current().start()