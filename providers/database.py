import pickle
import os

import numpy as np

class Database:
    def __init__(self, filename):
        self.data = {}  # Dictionary: {img_path: features}
        self.feature_matrix = None  # Precomputed feature matrix for fast querying
        self.img_paths = []  # Ordered list of image paths
        self.filename = filename
        self.save_after_every_addition = False

    def exists(self, img_path):
        """ Check if an image already exists in the database. """
        return img_path in self.data

    def add(self, feature_vector, img_path):
        """ Adds or updates an entry with features and associated image path. """
        if self.exists(img_path):
            print(f"Image '{img_path}' already exists. Updating features.")

        self.data[img_path] = np.array(feature_vector, dtype="float32")  # Store in dictionary
        self._update_feature_matrix()  # Update precomputed matrix
        if self.save_after_every_addition:
            self.save()

    def _update_feature_matrix(self):
        """ Recomputes the NumPy feature matrix for fast queries. """
        if self.data:
            self.feature_matrix = np.vstack(list(self.data.values()))  # Stack all feature vectors
            self.img_paths = list(self.data.keys())  # Maintain an ordered list of image paths
        else:
            self.feature_matrix = None
            self.img_paths = []

    def save(self):
        """ Saves the database to a file. """
        with open(self.filename, "wb") as f:
            pickle.dump(self.data, f)

    def load(self):
        """ Loads the database from a file and updates the feature matrix. """
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.data = pickle.load(f)
            self._update_feature_matrix()  # Precompute feature matrix on load
        else:
            print("No database file found. Starting with an empty database.")

    def query(self, query_vector, top_k=5):
        """ Finds the closest feature matches and returns (image_path, distance). """
        if self.feature_matrix is None:
            print("Database is empty. No query can be performed.")
            return []

        # Compute distances using precomputed feature matrix
        dists = np.linalg.norm(self.feature_matrix - query_vector, axis=1)

        # Get the `top_k` closest matches
        ids = np.argsort(dists)[:top_k]

        nearest_images = [{"image": str(self.img_paths[i]), "distance": float(dists[i])} for i in ids]

        return nearest_images
