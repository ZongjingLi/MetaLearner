'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-25 15:14:07
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-26 08:39:12
 # @ Description: This file is distributed under the MIT license.
'''

import tornado.ioloop
import tornado.web
import tornado.websocket
import random
import json

# 生成随机粒子的位置和速度
def generate_particle():
    return {
        "x": random.randint(0, 800),
        "y": random.randint(0, 600),
        "speedX": random.uniform(-1, 1),
        "speedY": random.uniform(-1, 1)
    }

particles = [generate_particle() for _ in range(100)]

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class ParticleWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        self.write_message(json.dumps({"particles": particles}))

    def on_message(self, message):
        pass

    def on_close(self):
        pass

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/particles", ParticleWebSocket),
        (r"/static/(.*)", tornado.web.StaticFileHandler, {"path": "static"}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8889)
    tornado.ioloop.IOLoop.current().start()
