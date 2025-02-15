import tensorflow_hub
import numpy as np
import cv2
import tf_keras

class ResNetExtractor:
    def __init__(self):
        layer = "https://www.kaggle.com/models/google/resnet-v2/TensorFlow2/152-feature-vector/2"
        #layer = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

        self.model = tf_keras.Sequential([
            tensorflow_hub.KerasLayer(layer, trainable=False)
        ])
        self.model.build([None, 224, 224, 3])  # Batch input shape.

    def extract(self, image_path):
        if isinstance(image_path, np.ndarray):
            img = image_path  # It's already an image, no need to load
        else:
            img = cv2.imread(image_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (224,224))

        img = tf_keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(img, axis=0) # Add a dimension

        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)  # Normalize
