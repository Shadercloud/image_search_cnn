class CompactHandler:
    """
    GET /compact?confirm=1 starts folding the delta log into a new base index in the
    background (see Database.compact). It rewrites the whole ~30G index, rate-limited by
    faiss.save_max_mb_per_sec, and adds wait until it finishes, so pause them first and
    run it at a quiet time. Progress: GET /stats ("compacting", "generation", "delta").
    """

    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.database = database

    def handle(self, query_params):
        confirm = query_params.get("confirm")
        if isinstance(confirm, list):
            confirm = confirm[0]
        if str(confirm) != "1":
            return self.request.json({"error": "Compaction rewrites the whole index (~30G of writes). "
                                               "Call /compact?confirm=1 to start it.",
                                      **self.database.status()})

        started = self.database.start_compaction()
        return self.request.json({"started": started, **self.database.status()})
