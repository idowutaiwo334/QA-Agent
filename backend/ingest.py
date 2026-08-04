"""
Run this whenever you add or change files in /data:

    python ingest.py

It rebuilds the vector database from scratch using everything in /data.
"""

from rag import ingest

if __name__ == "__main__":
    ingest()
