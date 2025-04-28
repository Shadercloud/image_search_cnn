from PIL import Image
from helpers.image_helper import readb64
from pathlib import Path
import cv2

class RotationHandler:
    def __init__(self, program_args, request, feature_extractor, database, shutdown_event):
        self.request = request
        self.feature_extractor = feature_extractor
        self.database = database
        self.verbose = program_args.verbose

    def handle(self, query_params):
        if "image" not in query_params:
            return self.request.json({"error": "Missing 'image' parameter"})

        if isinstance(query_params["image"], str):
            file = readb64(query_params["image"])
        else:
            image_value = query_params["image"][0]
            p = Path(image_value)

            if not p.is_file():
                return self.request.json({"error": f"File does not exist: {image_value}"})
            file = cv2.imread(p.resolve())

        file = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        file = Image.fromarray(file)

        if self.verbose > 1:
            print("Getting image rotation")
        rotation = self.feature_extractor.predict_rotation(file)
        if self.verbose > 1:
            print(f"Received Rotation of {rotation}")

        return self.request.json({"rotation": rotation})
