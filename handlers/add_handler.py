import os
from pathlib import Path

def get_image_files(directory):
    valid_extensions = {".jpg", ".jpeg", ".png"}
    files = []

    with os.scandir(directory) as entries:  # Much faster than Path.iterdir()
        for entry in entries:
            if entry.is_file() and Path(entry.name).suffix.lower() in valid_extensions:
                files.append(Path(entry.path).resolve())

    return files


class AddHandler:
    def __init__(self, request, feature_extractor, database):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database

    def handle(self, query_params):
        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        image_value = query_params["image"][0]
        p = Path(image_value)
        files = []

        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            files.append(image_value)
        elif p.is_dir():
            files = get_image_files(p)
            #files = [file.resolve() for file in p.iterdir() if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        elif not p.exists():
            return self.request.json({"error": f"File does not exist: {image_value}"})

        max_files = 1000000
        if "limit" in query_params:
            max_files = int(query_params["limit"][0])

        print(f"{len(files)} files found (processing {max_files})")

        done = 0
        for i, file in enumerate(files):
            if self.database.exists(file):
                print(f"Image {file} already exists")
                continue
            features = self.feature_extractor.extract(file)
            self.database.add(features, file)
            print(f"Done Image {i}: {file}")
            done += 1
            if self.database.save_after_additions > 0 and done % self.database.save_after_additions == 0:
                self.database.save()

            if done >= max_files:
                print(f"File Limit of {max_files} exceeded")
                break

        self.database.save()
        return self.request.json({"message": "Extracted features from images", "count": done})
