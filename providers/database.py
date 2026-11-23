import glob
import json
import os
import re
import sqlite3
from pathlib import Path
import faiss
import numpy as np
import time
import threading
from helpers.config_helper import config
from providers.delta_log import DeltaLog, _fsync_dir

# Mapping entry for an index vector whose image name is unknown (see _fill_mapping_gap)
UNNAMED = "__unnamed__"


class Database:
    """
    Feature vectors in sqlite plus an in-memory FAISS index for search.

    On disk the index is a *base* (FAISS index + name mapping, written rarely) plus an
    append-only *delta log* of everything added since (see providers/delta_log.py).
    save() only fsyncs the delta log; compact() folds the delta into a new base and is
    the only operation that rewrites the whole ~30G index. Bases are numbered
    generations, and the `<stem>_faiss_index.current.json` pointer names the live one.
    Replacing that pointer is the commit step, so an interrupted compaction leaves the
    previous base and its delta intact. Generation 0 is the original pair of files
    without a generation number in their names.
    """

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
        # (base mapping array, names added since as a list). Index position N belongs to
        # base[N] for N < len(base), else to added[N - len(base)]. Kept as one tuple so
        # a search never sees a new base with an old list, or the other way round.
        self._names = (np.array([], dtype=str), [])
        # File names in the index, for O(1) membership checks:
        # `name in mapping` scans the whole array (millions of entries) every time
        self.names = set()
        self.verbose = 0
        self.batch_size = config.get("faiss", "batch_size", default=1000000)
        self.vector_dim = config.get("faiss", "vector_dim", default=768)
        self.nprobe = config.get("faiss", "nprobe", default=50)
        self.nlist = config.get("faiss", "nlist", default=4096)
        self.data_dir = str(Path(filename).parent)
        self.stem = Path(filename).stem
        self.pointer_path = os.path.join(self.data_dir, f"{self.stem}_faiss_index.current.json")
        self.generation = 0
        self.delta = None
        self.compacting = False
        # Held by add() and compact(); searches never take it
        self.lock = threading.Lock()

    # ---- file layout -------------------------------------------------------------

    def _base_paths(self, generation):
        if generation == 0:
            return (os.path.join(self.data_dir, f"{self.stem}_faiss_index.ivf"),
                    os.path.join(self.data_dir, f"{self.stem}_faiss_index_mapping.npy"))
        return (os.path.join(self.data_dir, f"{self.stem}_faiss_index.g{generation}.ivf"),
                os.path.join(self.data_dir, f"{self.stem}_faiss_index_mapping.g{generation}.npy"))

    def _delta_path(self, generation):
        return os.path.join(self.data_dir, f"{self.stem}_faiss_index_delta.g{generation}.log")

    def _current_generation(self):
        """The live base generation, or None if there is no index at all yet."""
        if os.path.exists(self.pointer_path):
            with open(self.pointer_path) as f:
                return int(json.load(f)["generation"])
        index_path, mapping_path = self._base_paths(0)
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            return 0
        return None

    # ---- sqlite ------------------------------------------------------------------

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
        return img_file in self.names

    def missing(self, img_files):
        """ The file names (basenames) from img_files that are not in the index. """
        return [f for f in img_files if os.path.basename(f) not in self.names]

    def _load_names(self):
        base, added = self._names
        self.names = set(base.tolist())
        self.names.update(added)
        self.names.discard(UNNAMED)
        print(f"[INFO] Name set built with {len(self.names)} entries")

    # ---- adding ------------------------------------------------------------------

    def add(self, feature_vector, img_path):
        """ Adds or updates an entry with features and associated image path. """
        basename = os.path.basename(img_path)
        feature_vector = np.ascontiguousarray(feature_vector, dtype=np.float32).reshape(-1)
        feature_blob = feature_vector.tobytes()

        with self.conn:
            if self.exists(basename):
                print(f"Image '{basename}' already exists. Updating features.")
                self.conn.execute("UPDATE images SET features = ? WHERE img_file = ?", (feature_blob, basename))
            else:
                self.conn.execute("REPLACE INTO images (img_file, features) VALUES (?, ?)", (basename, feature_blob))

        with self.lock:
            # Log first: if the process dies after this, the add is replayed on the next
            # start; nothing is ever in the index without a name next to it on disk.
            self.delta.append(basename, feature_vector)
            # Name before vector, so a concurrent search never finds a vector whose
            # name is not there yet
            self._names[1].append(basename)
            self.index.add(np.expand_dims(feature_vector, axis=0))
            self.names.add(basename)

    # ---- loading -----------------------------------------------------------------

    def load(self):
        generation = self._current_generation()
        if generation is None:
            print("[INFO] No FAISS index found, so building a new one")
            self.build()
            return

        self.generation = generation
        index_path, mapping_path = self._base_paths(generation)
        print(f"[INFO] Loading base generation {generation}: {os.path.basename(index_path)}")

        print("[INFO] Loading index mapping...")
        mapping = np.load(mapping_path, allow_pickle=True)
        print(f"[INFO] Mapping loaded with {mapping.shape[0]} indexes")

        print("[INFO] Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = self.nprobe

        self._names = (mapping, [])
        self._reconcile_index_and_mapping()
        self._replay_delta()
        self._load_names()
        self._remove_stale_files()

        print("[INFO] Loading Database Completed")

    def _replay_delta(self):
        self.delta = DeltaLog(self._delta_path(self.generation), self.vector_dim)
        names, vectors = self.delta.read()
        if not names:
            print("[INFO] Delta log is empty")
            return
        started = time.time()
        for start in range(0, len(names), 100000):
            self.index.add(vectors[start:start + 100000])
        self._names[1].extend(names)
        print(f"[INFO] Replayed {len(names)} vectors from the delta log in {time.time() - started:.1f}s")

    def load_vector_batch(self, total, batch_size, random=False):

        for offset in range(0, total, batch_size):
            start_time = time.time()
            if random:
                query = f"""
                            SELECT img_file, features
                            FROM images
                            ORDER BY RANDOM()
                            LIMIT {batch_size}
                        """
            else:
                query = f"""
                            SELECT img_file, features
                            FROM images
                            LIMIT {batch_size} OFFSET {offset}
                        """

            self.cursor.execute(query)
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
        train_batch, _ = next(self.load_vector_batch(total, self.batch_size, True))
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

        self._names = (np.array(all_img_files), [])
        self.generation = 0
        self._write_base(1)
        self.delta = DeltaLog(self._delta_path(self.generation), self.vector_dim)
        self.delta.read()
        self._load_names()

        print("[INFO] Index and mapping saved successfully.")

    # ---- searching ---------------------------------------------------------------

    def _name_at(self, idx, names):
        base, added = names
        return base[idx] if idx < base.shape[0] else added[idx - base.shape[0]]

    def query(self, query_vector, top_k=5):
        query = np.expand_dims(query_vector, axis=0).astype('float32')

        names = self._names
        d, i = self.index.search(query, top_k)
        results = []
        for rank, (idx, dist) in enumerate(zip(i[0], d[0])):
            if idx == -1:
                continue
            img_file = self._name_at(int(idx), names)
            if img_file == UNNAMED:
                continue

            results.append({
                "image": str(img_file),
                "distance": float(dist)
            })

        return results

    def count(self):
        base, added = self._names
        return base.shape[0] + len(added)

    def status(self):
        return {
            "images": self.count(),
            "generation": self.generation,
            "delta": self.delta.records if self.delta else 0,
            "compacting": self.compacting,
        }

    def remove(self, img_path):
        basename = os.path.basename(img_path)

        with self.conn:
            if self.exists(basename):
                self.conn.execute("DELETE FROM images WHERE img_file = ?", (basename, ))
                return True

        return False

    # ---- repairs of an inconsistent base (in memory only) ------------------------

    def _reconcile_index_and_mapping(self):
        """
        Index position N belongs to mapping[N], so the two must have the same length.
        A save interrupted between writing the index and the mapping leaves them out of
        step, and every later add() would then be filed under the wrong name. Trim
        whichever is longer back to the shorter one. This happens in memory only: the
        next compaction writes a consistent base.
        """
        self._fill_mapping_gap()
        mapping = self._names[0]
        n_index, n_mapping = self.index.ntotal, mapping.shape[0]
        if n_index == n_mapping:
            return
        print(f"[WARN] FAISS index has {n_index} vectors but mapping has {n_mapping} names - repairing")
        if n_index > n_mapping:
            # add() gives vectors sequential ids, so the extras are ids n_mapping..n_index-1
            removed = self.index.remove_ids(faiss.IDSelectorRange(n_mapping, n_index))
            print(f"[WARN] Dropped {removed} index vectors with no name; they are still in "
                  f"the sqlite database and are re-added when their images are added again")
        else:
            mapping = mapping[:n_index]
            self._names = (mapping, [])
            print(f"[WARN] Dropped {n_mapping - n_index} mapping names with no vector")
        if self.index.ntotal != self._names[0].shape[0]:
            raise RuntimeError(f"Index/mapping repair failed: {self.index.ntotal} vs {self._names[0].shape[0]}")

    def _fill_mapping_gap(self):
        """
        One-time correction for vectors that were added to the index without a name.
        Every vector after them then belongs to the name `count` places earlier in the
        mapping, so searches return the wrong image. Configured as
            faiss:
              mapping_gap: {at: <first unnamed index position>, count: <how many>}
        Inserts `count` placeholder names at `at`, in memory only. It is skipped once the
        mapping already has the placeholders there (after the next compaction), so the
        setting can be removed after that.
        """
        gap = config.get("faiss", "mapping_gap", default=None)
        if not gap:
            return
        at, count = int(gap["at"]), int(gap["count"])
        mapping = self._names[0]
        if all(str(x) == UNNAMED for x in mapping[at:at + count]) and at + count <= mapping.shape[0]:
            print(f"[INFO] mapping_gap at {at} already filled - the setting can be removed")
            return
        if self.index.ntotal < mapping.shape[0] + count:
            print(f"[WARN] mapping_gap ignored: index has {self.index.ntotal} vectors, too few "
                  f"for {mapping.shape[0]} names plus {count} placeholders")
            return
        placeholders = np.full(count, UNNAMED, dtype=np.result_type(mapping.dtype, np.array([UNNAMED]).dtype))
        self._names = (np.concatenate([mapping[:at], placeholders, mapping[at:]]), [])
        print(f"[WARN] Inserted {count} placeholder names at position {at} (vectors with no name)")

    # ---- saving ------------------------------------------------------------------

    def save(self):
        """Make every add so far durable. Cheap: only the delta log is fsynced."""
        if self.delta is not None and self.delta.sync():
            print(f"[INFO] Delta log synced ({self.delta.records} records since generation {self.generation})")

    def start_compaction(self):
        """Run compact() in a background thread. Returns False if one is already running."""
        with self.lock:
            if self.compacting:
                return False
            self.compacting = True
        # Not a daemon thread, so a shutdown waits for it to finish
        threading.Thread(target=self._compact_thread, name="compaction").start()
        return True

    def _compact_thread(self):
        try:
            self.compact()
        except Exception as e:
            print(f"[ERROR] Compaction failed, the previous base and delta log are unchanged: {e}")
        finally:
            self.compacting = False

    def compact(self):
        """
        Write the whole in-memory index as a new base generation and start an empty delta
        log. This is the only operation that rewrites the full ~30G index (rate-limited by
        faiss.save_max_mb_per_sec), so run it deliberately at a quiet time. Adds wait for
        it to finish, so pause them first. Searches keep working throughout.
        """
        with self.lock:
            old_generation = self.generation
            new_generation = old_generation + 1
            print(f"[INFO] Compacting generation {old_generation} + {self.delta.records} delta "
                  f"records into generation {new_generation}...")
            started = time.time()
            self._write_base(new_generation)
            # Committed: the old delta's records are now part of the new base
            self.delta.close()
            self.delta = DeltaLog(self._delta_path(new_generation), self.vector_dim)
            self.delta.read()
            print(f"[INFO] Compaction to generation {new_generation} completed in {time.time() - started:.0f}s")
        self._remove_stale_files()

    def _write_base(self, generation):
        """Write the in-memory index and names as base `generation` and commit it."""
        base, added = self._names
        # concatenate widens the fixed-width string dtype if an added name is longer
        mapping = np.concatenate([base, np.array(added)]) if added else base
        if mapping.shape[0] != self.index.ntotal:
            raise RuntimeError(f"refusing to write a base with {self.index.ntotal} vectors "
                               f"but {mapping.shape[0]} names")
        index_path, mapping_path = self._base_paths(generation)

        print(f"[INFO] Writing {os.path.basename(index_path)}...")
        self._write_index_throttled(index_path + ".tmp")
        with open(mapping_path + ".tmp", "wb") as f:
            np.save(f, mapping)
            f.flush()
            os.fsync(f.fileno())
        os.replace(index_path + ".tmp", index_path)
        os.replace(mapping_path + ".tmp", mapping_path)

        pointer_tmp = self.pointer_path + ".tmp"
        with open(pointer_tmp, "w") as f:
            json.dump({"generation": generation, "vectors": int(self.index.ntotal),
                       "written_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(pointer_tmp, self.pointer_path)
        _fsync_dir(self.data_dir)

        self.generation = generation
        self._names = (mapping, [])

    def _remove_stale_files(self):
        """
        Delete what the live generation no longer needs: delta logs of older generations
        (already folded into the base), bases older than the previous one (the previous
        base is kept as a fallback), and leftovers of an interrupted compaction.
        """
        live = self.generation
        stale = []
        for path in glob.glob(os.path.join(self.data_dir, f"{self.stem}_faiss_index*")):
            name = os.path.basename(path)
            if name.endswith(".tmp"):
                stale.append(path)
                continue
            m = re.search(r"\.g(\d+)\.(ivf|npy|log)$", name)
            generation = int(m.group(1)) if m else (0 if name.endswith((".ivf", ".npy")) else None)
            if generation is None:
                continue
            if name.endswith(".log") and generation != live:
                stale.append(path)
            elif name.endswith((".ivf", ".npy")) and (generation < live - 1 or generation > live):
                stale.append(path)
        for path in stale:
            print(f"[INFO] Removing stale index file {os.path.basename(path)}")
            os.remove(path)

    def _write_index_throttled(self, path):
        """
        faiss.write_index, optionally limited to faiss.save_max_mb_per_sec (0 = unlimited).
        A full write is the whole index (~30G), which at full speed can saturate slow SSD
        storage and stall everything else on the host.
        """
        max_mb_per_sec = config.get("faiss", "save_max_mb_per_sec", default=0) or 0
        if not max_mb_per_sec:
            faiss.write_index(self.index, path)
            with open(path, "rb+") as f:
                os.fsync(f.fileno())
            return

        max_bytes_per_sec = max_mb_per_sec * 1024 * 1024
        with open(path, "wb") as f:
            started = time.monotonic()
            written = 0

            def write(chunk):
                nonlocal written
                f.write(chunk)
                written += len(chunk)
                ahead = written / max_bytes_per_sec - (time.monotonic() - started)
                if ahead > 0:
                    time.sleep(ahead)
                return len(chunk)

            faiss.write_index(self.index, faiss.PyCallbackIOWriter(write, 8 * 1024 * 1024))
            f.flush()
            os.fsync(f.fileno())
