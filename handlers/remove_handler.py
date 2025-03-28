class RemoveHandler:
    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.database = database

    def handle(self, query_params):
        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        if self.database.remove(query_params["image"][0]):
            return self.request.json({"message": "image has been deleted"})

        return self.request.json({"error": "Image does not exist"})
