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
parser.add_argument(
    "--safe",
    action="store_true",
    help="Write the data to the database file after every image file"
)
args = parser.parse_args()

# Dynamically import the chosen extractor
extractor_module = importlib.import_module(f"extractors.{args.extractor}_extractor")
FeatureExtractor = getattr(extractor_module, f"{args.extractor.capitalize()}Extractor")


# Initialize the selected feature extractor
feature_extractor = FeatureExtractor()

# Initialize the database
database = Database(Path(__file__).parent.resolve() / "data" / f"{args.extractor}.pickle")
database.load()
database.save_after_every_addition = args.safe

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
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

        routes = {
            "/add": AddHandler,
            "/search": SearchHandler,
            "/stats": StatsHandler
        }

        handler_class = routes.get(parsed_url.path)

        if handler_class:
            handler_class(self, feature_extractor, database).handle(query_params)
        else:
            self.not_found()

if __name__ == "__main__":
    webServer = WebServer(SimpleHTTPRequestHandler)
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
