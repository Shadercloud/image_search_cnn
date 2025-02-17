class StatsHandler:
    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database

    def handle(self, query_params):

        return self.request.json({"images": self.database.count()})
