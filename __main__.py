import argparse
import importlib
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import sleep
from urllib.parse import urlparse, parse_qs
import orjson
from handlers.stats_handler import StatsHandler
from providers.database import Database
from providers.webserver import WebServer

from handlers.add_handler import AddHandler
from handlers.search_handler import SearchHandler

shutdown_event = threading.Event()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the feature extraction web server.")
parser.add_argument("--extractor",default="clip", type=str, choices=["clip", "resnet"], required=False, help="Choose which feature extractor to use (clip or resnet)")
parser.add_argument("--host",default="localhost", type=str, required=False, help="Webserver host")
parser.add_argument("--port",default=8080, type=int, required=False, help="Webserver port")
parser.add_argument("--verbose", "-v", dest="verbose", default=0, type=int, required=False, help="Level of log output (0 = Not much, 1 = Info, 2 = Debug")
parser.add_argument("--output", dest="output", default=100, type=int, required=False, help="Output the count log after processing this many images.")
parser.add_argument("--update",dest="update_flag", action="store_true", help="Update an image if it already exists, instead of skipping it. (Default False")
parser.add_argument("--threads", dest="threads", default=2, type=int, required=False, help="Number of threads to run during image processing.")
params_args = parser.parse_args()

# Dynamically import the chosen extractor
extractor_module = importlib.import_module(f"extractors.{params_args.extractor}_extractor")
FeatureExtractor = getattr(extractor_module, f"{params_args.extractor.capitalize()}Extractor")


# Initialize the selected feature extractor
feature_extractor = FeatureExtractor()

# Initialize the database
database = Database(Path(__file__).parent.resolve() / "data" / f"{params_args.extractor}")
database.load()
database.verbose = params_args.verbose


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        """ Initialize request handler and define routes. """
        self.routes = {
            "/add": AddHandler,
            "/search": SearchHandler,
            "/stats": StatsHandler
        }
        super().__init__(*args, **kwargs)  # Call parent constructor

    def json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(orjson.dumps(data))

    def not_found(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(orjson.dumps({"error": "Not Found"}))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        handler_class = self.routes.get(parsed_url.path)

        if handler_class:
            handler_class(params_args, self, feature_extractor, database, shutdown_event).handle(query_params)
        else:
            self.not_found()

    def do_POST(self):
        if params_args.verbose > 1:
            print(f"Received POST request: {self.path}")
        """ Handles POST requests by parsing JSON data and routing accordingly. """
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        if params_args.verbose > 1:
            print("Finished receiving post data")

        try:
            json_data = orjson.loads(post_data)  # Parse JSON data
        except orjson.JSONDecodeError:
            return self.json({"error": "Invalid JSON"})

        parsed_url = urlparse(self.path)

        if params_args.verbose > 1:
            print(f"Loading request handler for {parsed_url.path}")

        handler_class = self.routes.get(parsed_url.path)

        if handler_class:
            handler_class(params_args, self, feature_extractor, database, shutdown_event).handle(json_data)
        else:
            self.not_found()

if __name__ == "__main__":
    webServer = WebServer(SimpleHTTPRequestHandler, params_args.host, params_args.port)
    webServer.start()
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            print('Keyboard Interrupt sent.')
            shutdown_event.set()
            sleep(2)
            webServer.shutdown()
            print("Saving Database")
            print("Exiting")
            exit(0)
