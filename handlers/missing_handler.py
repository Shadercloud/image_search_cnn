class MissingHandler:
    """
    POST /missing {"images": ["a.jpg", "dir/b.jpg", ...]}
    -> {"missing": [...]} - the given entries whose file name is not in the index.

    Lets a client find images that were never indexed (in batches) without sending
    each one to /add.
    """
    MAX_IMAGES = 10000

    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.database = database

    def handle(self, params):
        images = params.get("images") if isinstance(params, dict) else None
        if not isinstance(images, list):
            return self.request.json({"error": "POST a JSON body with an 'images' list"})
        if len(images) > self.MAX_IMAGES:
            return self.request.json({"error": f"At most {self.MAX_IMAGES} images per request"})

        return self.request.json({"missing": self.database.missing([str(i) for i in images])})
