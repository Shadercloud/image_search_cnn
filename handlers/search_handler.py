from pathlib import Path

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
            limit = int(query_params["limit"])

        image_value = query_params["image"][0]
        p = Path(image_value)

        if not p.is_file():
            return self.request.json({"error": f"File does not exist: {image_value}"})

        features = self.feature_extractor.extract(p.resolve())
        results = self.database.query(features, limit)

        return self.request.json({"results": results})
