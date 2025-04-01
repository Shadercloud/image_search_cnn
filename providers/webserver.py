import threading
from http.server import ThreadingHTTPServer

from helpers.config_helper import config


class WebServer(threading.Thread):
    def __init__(self, handler, host="localhost", port=8080):
        super().__init__()
        self.host = host
        self.port = port
        self.ws = ThreadingHTTPServer((self.host, self.port), handler)
        self.shutdown_event = threading.Event()
        self.periodic_callbacks = []

    def run(self):
        periodic_thread = threading.Thread(target=self.periodic_task)
        periodic_thread.daemon = True
        periodic_thread.start()

        print(f"Server started at http://{self.host}:{self.port}")

        self.ws.serve_forever()

    def add_task(self, task):
        if callable(task):
            print("[INFO] Adding periodic callback task")
            self.periodic_callbacks.append(task)
        else:
            raise TypeError("Callback must be callable")

    def periodic_task(self):
        wait_seconds = config.get("webserver", "periodic_task", default=300)
        while not self.shutdown_event.is_set():
            if self.shutdown_event.wait(timeout=wait_seconds):  # 300 seconds = 5 minutes
                break  # Exit if shutdown
            # --- Do your periodic action here ---
            print(f"[INFO] {wait_seconds}-second periodic task running...")
            for callback in self.periodic_callbacks:
                try:
                    callback()
                except Exception as e:
                    print(f"[ERROR] Periodic callback error: {e}")

    def shutdown(self):
        self.shutdown_event.set()
        print('[INFO] Shutting down server.')
        # call it anyway, for good measure...
        self.ws.shutdown()
        print('[INFO] Closing server.')
        self.ws.server_close()
        print('[INFO] Closing thread.')
        self.join()
