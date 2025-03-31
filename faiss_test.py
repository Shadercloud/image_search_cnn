import sqlite3
import numpy as np
import faiss
import time

faiss.omp_set_num_threads(faiss.omp_get_max_threads() - 2)

# =============================
# CONFIG
# =============================

SQLITE_DB_PATH = "data/clip.sqlite"
TABLE_NAME = "images"
OUTPUT_INDEX_PATH = "data/faiss_index.ivf"
OUTPUT_MAPPING_PATH = "data/faiss_index_mapping.npy"
BATCH_SIZE = 1000000   # adjust based on RAM
NLIST = 4096           # number of clusters
VECTOR_DIM = 768       # CLIP ViT-L/14 output

# =============================
# DATA LOADER
# =============================

def load_vectors(batch_size):
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
    total = cursor.fetchone()[0]

    for offset in range(0, total, batch_size):
        start_time = time.time()
        cursor.execute(f"""
            SELECT img_file, features
            FROM {TABLE_NAME}
            LIMIT {batch_size} OFFSET {offset}
        """)
        rows = cursor.fetchall()

        img_files = []
        vectors = []

        for img_file, feature_blob in rows:
            feature_vector = np.frombuffer(feature_blob, dtype=np.float32)
            img_files.append(img_file)
            vectors.append(feature_vector)

        elapsed = time.time() - start_time
        print(f"[INFO] Loaded batch: offset={offset}, size={len(vectors)}, time={elapsed:.2f}s")

        yield np.vstack(vectors), img_files

    conn.close()

# =============================
# FAISS INDEX BUILD
# =============================

def build_faiss_index():
    print(f"[INFO] Number of threads: {faiss.omp_get_max_threads()}")

    # ---- TRAINING ----
    print("[INFO] Loading training batch...")
    train_batch, _ = next(load_vectors(BATCH_SIZE))
    print(f"[INFO] Training index on {train_batch.shape[0]} vectors...")

    # Prepare index
    quantizer = faiss.IndexFlatL2(VECTOR_DIM)
    cpu_index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, NLIST, faiss.METRIC_L2)

    cpu_index.train(train_batch)
    print("[INFO] Training done on CPU.")

    # ---- ADD VECTORS + BUILD MAPPING ----
    print("[INFO] Adding vectors on CPU...")
    all_img_files = []
    batch_num = 0
    total_added = 0
    start_time = time.time()

    for batch, img_files in load_vectors(BATCH_SIZE):
        cpu_index.add(batch)
        all_img_files.extend(img_files)
        total_added += len(batch)
        batch_num += 1
        print(f"[INFO] Added batch {batch_num}: {len(batch)} vectors, total added: {total_added}")

    elapsed = time.time() - start_time
    print(f"[INFO] Finished adding vectors. Total vectors: {total_added}, time: {elapsed:.2f}s")

    # ---- SAVE INDEX ----
    print(f"[INFO] Saving index to {OUTPUT_INDEX_PATH}")
    faiss.write_index(cpu_index, OUTPUT_INDEX_PATH)

    # ---- SAVE MAPPING ----
    print(f"[INFO] Saving ID mapping to {OUTPUT_MAPPING_PATH}")
    np.save(OUTPUT_MAPPING_PATH, np.array(all_img_files))

    print("[INFO] Index and mapping saved successfully.")

if __name__ == "__main__":
    build_faiss_index()
