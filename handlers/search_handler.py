from pathlib import Path

from helpers.image_helper import readb64
from providers.compare import Compare


class SearchHandler:
    def __init__(self, request, feature_extractor, database):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database

    def handle(self, query_params):
        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        limit = 10
        if "limit" in query_params:
            limit = int(query_params["limit"][0])

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
            file = p.resolve()

        features = self.feature_extractor.extract(file)

        results = self.database.query(features, limit)
        if compare_opts:
            comp = Compare(p.resolve())
            for result in results:
                comp.set(result['image'])
                result['compare'] = {}
                for c in compare_opts:
                    if hasattr(comp, c) and callable(getattr(comp, c)):
                        result['compare'][c] = str(getattr(comp, c)())

        return self.request.json({"results": results})
