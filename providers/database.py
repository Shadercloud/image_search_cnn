import os
import sqlite3
from pathlib import Path
import faiss
import numpy as np
import time
import threading
from helpers.config_helper import config

class Database:
    def __init__(self, filename):
        max_threads = config.get("cpu", "max_threads")
        if max_threads:
            faiss.omp_set_num_threads(max_threads)
            print(f"[INFO] Faiss max threads set to {max_threads}")

        self.filename = Path(filename).with_suffix(".sqlite")
        self.conn = sqlite3.connect(self.filename, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_table()
        self.index = None
        self.mapping = None
        self.verbose = 0
        self.batch_size = config.get("faiss", "batch_size", default=1000000)
        self.vector_dim = config.get("faiss", "vector_dim", default=768)
        self.nprobe = config.get("faiss", "nprobe", default=50)
        self.nlist = config.get("faiss", "nlist", default=4096)
        self.index_path = str(Path(filename).with_name(Path(filename).stem + "_faiss_index.ivf"))
        self.mapping_path = str(Path(filename).with_name(Path(filename).stem + "_faiss_index_mapping.npy"))
        self.index_changed = False
        self.lock = threading.Lock()

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
        return img_file in self.mapping

    def add(self, feature_vector, img_path):
        basename = os.path.basename(img_path)
        """ Adds or updates an entry with features and associated image path. """
        feature_blob = feature_vector.tobytes()

        with self.conn:
            if self.exists(basename):
                print(f"Image '{basename}' already exists. Updating features.")
                self.conn.execute("UPDATE images SET features = ? WHERE img_file = ?", (feature_blob, basename))
            else:
                self.conn.execute("REPLACE INTO images (img_file, features) VALUES (?, ?)", (basename, feature_blob))

        with self.lock:
            # Add vector to index
            self.index.add(np.expand_dims(feature_vector, axis=0))
            # Update mapping
            self.mapping = np.append(self.mapping, basename)

        self.index_changed = True


    def load(self):
        if Path(self.index_path).exists() and Path(self.mapping_path).exists():
            print("[INFO] Loading index mapping...")
            self.mapping = np.load(self.mapping_path, allow_pickle=True)

            print(f"[INFO] Mapping loaded with {self.mapping.shape[0]} indexes")

            print("[INFO] Loading FAISS index...")
            self.index = faiss.read_index(self.index_path)
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.nprobe

            print("[INFO] Loading Database Completed")
        else:
            print(f"[INFO] FAISS Index file {self.index_path} does not exists so build a new index")
            self.build()

    def load_vector_batch(self, total, batch_size):

        for offset in range(0, total, batch_size):
            start_time = time.time()
            self.cursor.execute(f"""
                SELECT img_file, features
                FROM images
                LIMIT {batch_size} OFFSET {offset}
            """)
            rows = self.cursor.fetchall()

            img_files = []
            vectors = []

            for img_file, feature_blob in rows:
                feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
                img_files.append(img_file)
                vectors.append(feature_vector)

            elapsed = time.time() - start_time
            print(f"[INFO] Loaded batch: offset={offset}, size={len(vectors)}, time={elapsed:.2f}s")

            yield np.vstack(vectors), img_files

    def build(self):
        self.cursor.execute("SELECT COUNT(*) FROM images")
        total = self.cursor.fetchone()[0]
        if total == 0:
            print("[INFO] Database does not have any data yet so cannot build indexes")
            return

        print(f"[INFO] Number of threads: {faiss.omp_get_max_threads()}")

        # ---- TRAINING ----
        print("[INFO] Loading training batch...")
        train_batch, _ = next(self.load_vector_batch(total, self.batch_size))
        print(f"[INFO] Training index on {train_batch.shape[0]} vectors...")

        # Prepare index
        quantizer = faiss.IndexFlatL2(self.vector_dim)
        self.index = faiss.IndexIVFFlat(quantizer, self.vector_dim, self.nlist, faiss.METRIC_L2)

        self.index.train(train_batch)
        print("[INFO] Training done on CPU.")

        # ---- ADD VECTORS + BUILD MAPPING ----
        print("[INFO] Adding vectors on CPU...")
        all_img_files = []
        batch_num = 0
        total_added = 0
        start_time = time.time()

        for batch, img_files in self.load_vector_batch(total, self.batch_size):
            self.index.add(batch)
            all_img_files.extend(img_files)
            total_added += len(batch)
            batch_num += 1
            print(f"[INFO] Added batch {batch_num}: {len(batch)} vectors, total added: {total_added}")

        elapsed = time.time() - start_time
        print(f"[INFO] Finished adding vectors. Total vectors: {total_added}, time: {elapsed:.2f}s")

        # ---- SAVE INDEX ----
        print(f"[INFO] Saving index to {self.index_path}")
        faiss.write_index(self.index, self.index_path)

        # ---- SAVE MAPPING ----
        print(f"[INFO] Saving ID mapping to {self.mapping_path}")
        self.mapping = np.array(all_img_files)
        np.save(self.mapping_path, self.mapping)

        print("[INFO] Index and mapping saved successfully.")

    def query(self, query_vector, top_k=5):
        query = np.expand_dims(query_vector, axis=0).astype('float32')

        d, i = self.index.search(query, top_k)
        results = []
        for rank, (idx, dist) in enumerate(zip(i[0], d[0])):
            if idx == -1:
                continue
            img_file = self.mapping[idx]

            results.append({
                "image": str(img_file),
                "distance": float(dist)
            })

        return results

    def count(self):
        return len(self.mapping)

    def remove(self, img_path):
        basename = os.path.basename(img_path)

        with self.conn:
            if self.exists(basename):
                self.conn.execute("DELETE FROM images WHERE img_file = ?", (basename, ))
                return True

        return False

    def save(self):
        if self.index_changed:
            with self.lock:
                print("[INFO] Saving FAISS index...")
                faiss.write_index(self.index, self.index_path)
                np.save(self.mapping_path, self.mapping)
        else:
            print("[INFO] No need to save FAISS index.")


