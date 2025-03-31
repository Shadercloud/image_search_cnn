import threading
from http.server import ThreadingHTTPServer


class WebServer(threading.Thread):
    def __init__(self, handler, host="localhost", port=8080):
        super().__init__()
        self.host = host
        self.port = port
        self.ws = ThreadingHTTPServer((self.host, self.port), handler)

    def run(self):
        print(f"Server started at http://{self.host}:{self.port}")
        self.ws.serve_forever()

    def shutdown(self):
        print('[INFO] Shutting down server.')
        # call it anyway, for good measure...
        self.ws.shutdown()
        print('[INFO] Closing server.')
        self.ws.server_close()
        print('[INFO] Closing thread.')
        self.join()
