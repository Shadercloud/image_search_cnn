from pathlib import Path
from helpers.image_helper import readb64
from providers.compare import Compare
import cv2

import yaml
config = yaml.safe_load(open(Path(__file__).parent.parent.resolve() / "config.yml"))


class SearchHandler:
    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database
        self.verbose = program_args.verbose

    def handle(self, query_params):
        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        limit = 10
        if "limit" in query_params:
            limit = int(isinstance(query_params["limit"], list) and query_params["limit"][0] or query_params["limit"])

        compare_opts = None
        if "compare" in query_params:
            compare_opts = query_params["compare"]

        if isinstance(query_params["image"], str):
            file = readb64(query_params["image"])
        else:
            image_value = query_params["image"][0]
            p = Path(image_value)

            if not p.is_file():
                return self.request.json({"error": f"File does not exist: {image_value}"})
            file = cv2.imread(p.resolve())

        if self.verbose > 1:
            print("Getting Image Features from CNN")
        features = self.feature_extractor.extract(file)
        if self.verbose > 1:
            print("Received Image Features from CNN")

        results = self.database.query(features, limit)
        if compare_opts and "images" in config and "path" in config["images"]:
            comp = Compare(file)
            for result in results:
                comp.set(Path(config['images']['path'] + result['image']))
                result['compare'] = {}
                for c in compare_opts:
                    if hasattr(comp, c) and callable(getattr(comp, c)):
                        result['compare'][c] = str(getattr(comp, c)())

        return self.request.json({"results": results})
