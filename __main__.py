import argparse
import importlib
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from time import sleep
from urllib.parse import urlparse, parse_qs
import json

from handlers.stats_handler import StatsHandler
from providers.database import Database
from providers.webserver import WebServer

from handlers.add_handler import AddHandler
from handlers.search_handler import SearchHandler

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the feature extraction web server.")
parser.add_argument("--extractor",default="clip", type=str, choices=["clip", "resnet"], required=False, help="Choose which feature extractor to use (clip or resnet)")
parser.add_argument("--host",default="localhost", type=str, required=False, help="Webserver host")
parser.add_argument("--port",default=8080, type=int, required=False, help="Webserver port")
parser.add_argument("--save",default=1, type=int, required=False, help="Write to the database after this many new additions (0 for only write after all adds are completed)")
args = parser.parse_args()

# Dynamically import the chosen extractor
extractor_module = importlib.import_module(f"extractors.{args.extractor}_extractor")
FeatureExtractor = getattr(extractor_module, f"{args.extractor.capitalize()}Extractor")


# Initialize the selected feature extractor
feature_extractor = FeatureExtractor()

# Initialize the database
database = Database(Path(__file__).parent.resolve() / "data" / f"{args.extractor}.pickle")
database.load()
database.save_after_additions = args.save

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
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def not_found(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_GET(self):
        parsed_url = urlparse(self.path)
        query_params = parse_qs(parsed_url.query)

        handler_class = self.routes.get(parsed_url.path)

        if handler_class:
            handler_class(self, feature_extractor, database).handle(query_params)
        else:
            self.not_found()

    def do_POST(self):
        """ Handles POST requests by parsing JSON data and routing accordingly. """
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            json_data = json.loads(post_data.decode("utf-8"))  # Parse JSON data
        except json.JSONDecodeError:
            return self.json({"error": "Invalid JSON"})

        parsed_url = urlparse(self.path)

        handler_class = self.routes.get(parsed_url.path)

        if handler_class:
            handler_class(self, feature_extractor, database).handle(json_data)
        else:
            self.not_found()

if __name__ == "__main__":
    webServer = WebServer(SimpleHTTPRequestHandler, args.host, args.port)
    webServer.start()
    while True:
        try:
            sleep(1)
        except KeyboardInterrupt:
            print('Keyboard Interrupt sent.')
            webServer.shutdown()
            print("Saving Database")
            database.save()
            print("Exiting")
            exit(0)
