import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

class AddHandler:
    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database
        self.allow_update = program_args.update_flag
        self.verbose = program_args.verbose
        self.output_count = program_args.output
        self.processed = 0
        self.skipped = 0
        self.lock = threading.Lock()
        self.max_files = 1000000
        self.queue = Queue()
        self.num_threads = program_args.threads
        self.shutdown_event = shutdown_event

    def process_image(self):
        """Worker function that processes images from the queue."""
        while True:
            image = self.queue.get()  # Get image path from queue
            if image is None:  # Sentinel value to signal exit
                break

            if self.shutdown_event.is_set():
                break

            if self.output_count > 0 and (self.skipped + self.processed) % self.output_count == 0:
                print(f"Completed {self.processed} | Skipped {self.skipped}")

            if not self.allow_update and self.database.exists(os.path.basename(image)):
                with self.lock:
                    self.skipped += 1
                if self.verbose > 1:
                    print(f"Image {image} already exists")
                continue

            features = self.feature_extractor.extract(image)
            if features is None:
                continue

            self.database.add(features, image)

            with self.lock:
                if self.processed >= self.max_files:
                    return
                self.processed += 1

            if self.verbose > 0:
                print(f"Done Image {self.processed}: {image}")


    def handle(self, query_params):
        valid_extensions = {".jpg", ".jpeg", ".png"}

        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        image_value = query_params["image"][0]
        p = Path(image_value)

        if "limit" in query_params:
            self.max_files = int(query_params["limit"][0])

        self.processed = 0
        self.skipped = 0

        if not p.exists():
            return self.request.json({"error": f"File does not exist: {image_value}"})

        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            self.queue.put(p.resolve())
        elif p.is_dir():
            # Start worker threads
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                print(f"Starting {self.num_threads} image processing threads")
                workers = [executor.submit(self.process_image) for _ in range(self.num_threads)]

                with os.scandir(p) as entries:
                    for entry in entries:
                        if entry.is_file() and Path(entry.name).suffix.lower() in valid_extensions:
                            self.queue.put(Path(entry.path).resolve())

                            if self.processed >= self.max_files:
                                print(f"File Limit of {self.max_files} exceeded")
                                break

                self.destroy_workers()

        print(f"Completed {self.processed} | Skipped {self.skipped}")

        return self.request.json({"message": "Extracted features from images", "added": self.processed, "skipped": self.skipped})

    def destroy_workers(self):
        for _ in range(self.num_threads):
            self.queue.put(None)

