import torch
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel

class ClipExtractor:
    def __init__(self):
        # Load CLIP model (ViT-L/14 for best accuracy)
        model_name = "openai/clip-vit-large-patch14"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def extract(self, image_path):
        # Load and preprocess image
        if isinstance(image_path, np.ndarray):
            img = image_path  # It's already an image, no need to load
        else:
            img = cv2.imread(image_path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (224, 224))  # Resize to 224x224

        # Preprocess for CLIP
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # Move to GPU if available

        # Extract feature vector
        with torch.no_grad():
            feature = self.model.get_image_features(**inputs)

        # Normalize feature vector
        feature = feature.cpu().numpy().flatten()
        return feature / np.linalg.norm(feature)  # L2 Normalize

