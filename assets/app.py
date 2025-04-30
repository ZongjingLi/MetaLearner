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
        # For demo purposes, we'll create a simple list
        items = []
        
        if input_text:
            items.extend(input_text.split())
            
        grounding = None
        if file_info:
            items.append(f"Processed file: {file_info['filename']}")
            if file_info['content_type'].startswith('image/'):
                items.append("Type: Image")
            elif file_info['content_type'].startswith('video/'):
                items.append("Type: Video")
                
            """TODO: the grounding of the model need to read the input and save here"""
            from torchvision import transforms
            from PIL import Image

            img = Image.open("assets"+file_info["path"])
            transform = transforms.ToTensor()
            tensor_img = transform(img)
            grounding = {"input_image" : tensor_img} # should not be jus a empty dict but actually read the image

        self.model(input_text, grounding)
        # Return results as JSON
        self.write(json.dumps({
            "list_items": items,
            "image_updated": True,
            "file_info": file_info
        }))

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