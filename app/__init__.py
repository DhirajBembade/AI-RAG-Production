import os

# faiss (OpenMP) and torch (bundles its own OpenMP runtime, pulled in by the
# sentence-transformers reranker) segfault when both are loaded into the same process
# on macOS — their bundled OpenMP runtimes collide. Must be set before either library
# is imported anywhere downstream; faiss is also pinned to 1 thread in faiss_store.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
