import os
import struct
import zlib

import numpy as np

MAGIC = b"FAISSDLT"
VERSION = 1
_HEADER = struct.Struct("<8sII")   # magic, format version, vector dimension
_RECORD = struct.Struct("<IH")     # crc32 of (name length + name + vector), name length
# Image names are short (sqlite column is VARCHAR(255)); the cap keeps "partial last
# record" distinguishable from damage further back in the file
MAX_NAME_BYTES = 1024


class DeltaLog:
    """
    Append-only log of the vectors added since the base index was written.

    Each record holds an image name together with its vector, so replaying the log can
    never pair a vector with the wrong name. A save only has to fsync this file (a few KB
    per add) instead of rewriting the whole ~30G index.
    """

    def __init__(self, path, dim):
        self.path = path
        self.dim = dim
        self.vector_bytes = dim * 4
        self.records = 0
        self.dirty = False
        self._file = None

    def read(self):
        """
        Return (names, vectors) for every complete record, creating an empty log if there
        is none. A record cut off by a crash at the very end of the file is dropped and
        the file truncated before it. Damage anywhere else raises instead, so later
        records are never discarded without anyone noticing.
        """
        names = []
        vectors = []
        if not os.path.exists(self.path):
            self._create()
            return names, np.empty((0, self.dim), dtype=np.float32)

        size = os.path.getsize(self.path)
        with open(self.path, "rb") as f:
            header = f.read(_HEADER.size)
            if len(header) < _HEADER.size:
                raise RuntimeError(f"{self.path}: truncated header")
            magic, version, dim = _HEADER.unpack(header)
            if magic != MAGIC or version != VERSION:
                raise RuntimeError(f"{self.path}: not a delta log (magic {magic!r}, version {version})")
            if dim != self.dim:
                raise RuntimeError(f"{self.path}: vector dimension {dim}, expected {self.dim}")

            good_end = f.tell()
            while good_end < size:
                head = f.read(_RECORD.size)
                if len(head) < _RECORD.size:
                    break
                crc, name_len = _RECORD.unpack(head)
                body = f.read(name_len + self.vector_bytes)
                if len(body) < name_len + self.vector_bytes:
                    break
                if zlib.crc32(head[4:] + body) != crc:
                    break
                names.append(body[:name_len].decode("utf-8"))
                vectors.append(np.frombuffer(body[name_len:], dtype=np.float32))
                good_end = f.tell()

        if good_end < size:
            torn = size - good_end
            # An interrupted append leaves at most one partial record
            if torn > _RECORD.size + MAX_NAME_BYTES + self.vector_bytes:
                raise RuntimeError(f"{self.path}: damaged record at byte {good_end} with {torn} bytes after it; "
                                   f"refusing to drop them - inspect the file")
            print(f"[WARN] {self.path}: dropping a {torn}-byte incomplete record at the end (interrupted write)")
            os.truncate(self.path, good_end)

        self.records = len(names)
        vectors = np.vstack(vectors) if vectors else np.empty((0, self.dim), dtype=np.float32)
        return names, vectors

    def append(self, name, vector):
        """Append one record. It reaches the OS at once and the disk at the next sync()."""
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > MAX_NAME_BYTES:
            raise ValueError(f"image name too long for the delta log: {name[:80]}...")
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"vector has {vector.shape[0]} dimensions, expected {self.dim}")
        body = struct.pack("<H", len(name_bytes)) + name_bytes + vector.tobytes()
        record = struct.pack("<I", zlib.crc32(body)) + body
        if self._file is None:
            self._file = open(self.path, "ab")
        self._file.write(record)
        self._file.flush()
        self.records += 1
        self.dirty = True

    def sync(self):
        """fsync appended records to disk. Returns True if there was anything to sync."""
        if not self.dirty or self._file is None:
            return False
        # Cleared before syncing: an append that lands meanwhile sets it again
        self.dirty = False
        self._file.flush()
        os.fsync(self._file.fileno())
        return True

    def close(self):
        if self._file is not None:
            self.sync()
            self._file.close()
            self._file = None

    def _create(self):
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(_HEADER.pack(MAGIC, VERSION, self.dim))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        _fsync_dir(os.path.dirname(self.path))
        self.records = 0


def _fsync_dir(path):
    fd = os.open(path or ".", os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
