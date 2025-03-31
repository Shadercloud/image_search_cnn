import faiss
import numpy as np

# =============================
# CONFIG
# =============================

INDEX_PATH = "data/faiss_index.ivf"
MAPPING_PATH = "data/faiss_index_mapping.npy"
NPROBE = 50  # You can tune this for accuracy/speed

# =============================
# LOADER
# =============================

def load_index_and_mapping():
    print("[INFO] Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    print(f"[INFO] Index loaded with {index.ntotal} vectors.")

    print("[INFO] Loading ID mapping...")
    mapping = np.load(MAPPING_PATH, allow_pickle=True)
    print(f"[INFO] Mapping loaded with {len(mapping)} entries.")

    if index.ntotal != len(mapping):
        print("[WARNING] Index size and mapping size do not match!")

    return index, mapping

# =============================
# UTILITY FUNCTIONS
# =============================

def show_index_info(index):
    print(f"[INFO] Index has {index.ntotal} vectors.")
    print(f"[INFO] Index dimension: {index.d}")
    if hasattr(index, "nlist"):
        print(f"[INFO] Index nlist (number of clusters): {index.nlist}")
    print(f"[INFO] nprobe (clusters searched per query): {index.nprobe}")

def search(index, mapping, query_vector, top_k=5):
    # Ensure query is 2D
    query = np.expand_dims(query_vector, axis=0).astype('float32')

    # Set nprobe
    if hasattr(index, "nprobe"):
        index.nprobe = NPROBE

    # Search
    D, I = index.search(query, top_k)
    print(f"[INFO] Search results (top {top_k}):")
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
        if idx == -1:
            continue
        img_file = mapping[idx]
        print(f"  {rank + 1}. {img_file} (distance: {dist:.4f})")
        results.append((img_file, dist))
    return results

# =============================
# EXAMPLE USAGE
# =============================

if __name__ == "__main__":
    # Load index & mapping
    index, mapping = load_index_and_mapping()

    # Show index info
    show_index_info(index)

    # Example: Search with random query
    print("[INFO] Running example search...")
    dummy_query = np.random.rand(index.d).astype('float32')
    dummy_query /= np.linalg.norm(dummy_query)  # normalize
    search(index, mapping, dummy_query, top_k=5)
