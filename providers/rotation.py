from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import MobileNet_V2_Weights

class RotationCNN:
    def __init__(self):
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        weights = MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.last_channel, 4)
        )

        # Load trained weights
        self.model.load_state_dict(torch.load(Path(__file__).parent.resolve().parent.resolve() / "models" / "rotation_classifier.pth",
                                              map_location=self.device,
                                              weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # Define transforms (use same as training)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            weights.transforms()  # Keep ImageNet normalization
        ])

        # Rotation labels
        self.rotation_labels = {
            0: "0",
            1: "180",
            2: "270",
            3: "90"
        }

    def predict_rotation(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)  # Add batch dimension

        # Predict
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()

        return self.rotation_labels[predicted_class]
