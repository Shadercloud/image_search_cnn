class StatsHandler:
    def __init__(self, request, feature_extractor, database):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database

    def handle(self, query_params):

        return self.request.json({"images": len(self.database.img_paths)})
