import os
import sqlite3
from pathlib import Path
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from sklearn.neighbors import NearestNeighbors
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
        self.nn_model = None
        self.verbose = 0

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

            self.nn_model = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
            self.nn_model.fit(self.feature_matrix)
        else:
            self.nn_model = None
            self.feature_matrix = None
            self.img_files = []


    def load(self, write_only=False):
        """ Loads all image features from the database. """
        self.cursor.execute("SELECT COUNT(*) FROM images")
        count = self.cursor.fetchone()[0]
        if write_only:
            print(f"Database is loading in WRITE ONLY mode, search queries will not work.  {count} images in the database.")
        else:
            print(f"Loading {count} images from the database.  This may take a while...")

        self.cursor.execute("SELECT img_file, features FROM images")
        for img_file, feature_blob in self.cursor.fetchall():
            if write_only:
                self.data[img_file] =np.array([1], dtype=np.float32) # Do this so that we can load the database with empty data and track the image keys only
            else:
                self.data[img_file] = np.frombuffer(feature_blob, dtype=np.float32)  # Convert binary back to NumPy array

        self._update_feature_matrix()

        print("Loading Database Completed")

    def query(self, query_vector, top_k=5):
        """ Finds the closest feature matches and returns (image_path, distance). """
        if self.features_altered:
            self._update_feature_matrix()

        if self.nn_model is None:
            print("Database is empty. No query can be performed.")
            return []

        if self.verbose > 1:
            print("Finding nearest neighbors...")

        dists, ids = self.nn_model.kneighbors([query_vector], n_neighbors=top_k)

        if self.verbose > 1:
            print(f"Found {len(ids[0])} nearest neighbors.")

        nearest_images = [{"image": self.img_files[ids[0][i]], "distance": float(dists[0][i])} for i in range(len(ids[0]))]

        return nearest_images

    def count(self):
        if self.features_altered:
            self._update_feature_matrix()
        return len(self.img_files)

    def remove(self, img_path):
        basename = os.path.basename(img_path)

        with self.conn:
            if self.exists(basename):
                self.conn.execute("DELETE FROM images WHERE img_file = ?", (basename, ))
                del self.data[basename]
                self.features_altered = True
                return True

        return False


