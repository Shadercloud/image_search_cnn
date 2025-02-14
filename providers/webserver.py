import threading
from http.server import HTTPServer


class WebServer(threading.Thread):
    def __init__(self, handler):
        super().__init__()
        self.host = "localhost"
        self.port = 8080
        self.ws = HTTPServer((self.host, self.port), handler)

    def run(self):
        print(f"Server started at http://{self.host}:{self.port}")
        self.ws.serve_forever()

    def shutdown(self):
        # set the two flags needed to shutdown the HTTP server manually
        self.ws._BaseServer__is_shut_down.set()
        self.ws.__shutdown_request = True

        print('Shutting down server.')
        # call it anyway, for good measure...
        self.ws.shutdown()
        print('Closing server.')
        self.ws.server_close()
        print('Closing thread.')
        self.join()
