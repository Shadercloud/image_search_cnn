import os
import sqlite3
from pathlib import Path

import numpy as np

class Database:
    def __init__(self, filename):
        self.data = {}  # Dictionary: {img_file: features}
        self.feature_matrix = None  # Precomputed feature matrix for fast querying
        self.features_altered = True
        self.img_files = []  # Ordered list of image paths
        self.filename = Path(filename).with_suffix(".sqlite")
        self.conn = sqlite3.connect(self.filename, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """ Creates table to store feature vectors. """
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                img_file VARCHAR(255) PRIMARY KEY,
                features BLOB
            )
        """)
        self.conn.commit()

    def exists(self, img_file):
        """ Check if an image already exists in the database. """
        return img_file in self.data

    def add(self, feature_vector, img_path):
        basename = os.path.basename(img_path)
        """ Adds or updates an entry with features and associated image path. """
        feature_blob = feature_vector.tobytes()

        with self.conn:
            if self.exists(basename):
                print(f"Image '{basename}' already exists. Updating features.")
                self.conn.execute("UPDATE images SET features = ? WHERE img_file = ?", (feature_blob, basename))
            else:
                self.conn.execute("INSERT INTO images (img_file, features) VALUES (?, ?)", (basename, feature_blob))

        self.data[basename] = np.array(feature_vector, dtype="float32")  # Store in dictionary
        self.features_altered = True

    def _update_feature_matrix(self):
        self.features_altered = False
        """ Recomputes the NumPy feature matrix for fast queries. """
        if self.data:
            self.feature_matrix = np.vstack(list(self.data.values()))  # Stack all feature vectors
            self.img_files = list(self.data.keys())  # Maintain an ordered list of image files
        else:
            self.feature_matrix = None
            self.img_files = []

    def load(self):
        """ Loads all image features from the database. """
        self.cursor.execute("SELECT COUNT(*) FROM images")
        count = self.cursor.fetchone()[0]
        print(f"Loading {count} images from the database.  This may take a while...")

        self.cursor.execute("SELECT img_file, features FROM images")
        for img_file, feature_blob in self.cursor.fetchall():
            self.data[img_file] = np.frombuffer(feature_blob, dtype=np.float32)  # Convert binary back to NumPy array

        self._update_feature_matrix()

        print("Loading Database Completed")

    def query(self, query_vector, top_k=5):
        """ Finds the closest feature matches and returns (image_path, distance). """
        if self.features_altered:
            self._update_feature_matrix()

        if self.feature_matrix is None:
            print("Database is empty. No query can be performed.")
            return []

        # Compute distances using precomputed feature matrix
        dists = np.linalg.norm(self.feature_matrix - query_vector, axis=1)

        # Get the `top_k` closest matches
        ids = np.argsort(dists)[:top_k]

        nearest_images = [{"image": str(self.img_files[i]), "distance": float(dists[i])} for i in ids]

        return nearest_images

    def count(self):
        return len(self.img_files)
