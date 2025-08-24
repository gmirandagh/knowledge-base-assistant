import os
import minsearch
import json
import pickle

PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = os.path.join(PROJECT_ROOT, 'data', 'data.jsonl')
data_index_path = os.path.join(PROJECT_ROOT, 'data', 'data_index.bin')


def build_index():
    """Build a new Minsearch index from data.jsonl and save it."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load processed data
    with open(data_path, 'rt', encoding='utf-8') as f_in:
        docs = [json.loads(line) for line in f_in]

    # Filter out any chunks where embedding failed
    documents = [doc for doc in docs if doc.get('embedding') is not None]
    print(f"Loaded {len(documents)} documents with embeddings.")

    # Create Minsearch index
    index = minsearch.Index(
        text_fields=["text"],
        keyword_fields=["title", "document_id", "chunk_type", "section_title"]
    )
    index.fit(documents)

    # Save the index
    with open(data_index_path, 'wb') as f_out:
        pickle.dump(index, f_out)

    print(f"\n✅ Minsearch index created and saved to '{data_index_path}'")
    return index


def load_index(force_rebuild: bool = False):
    """
    Load the Minsearch index. Rebuild if missing or if force_rebuild=True.
    """
    if not force_rebuild and os.path.exists(data_index_path):
        try:
            with open(data_index_path, 'rb') as f_in:
                index = pickle.load(f_in)
            print(f"✅ Loaded existing index from '{data_index_path}'")
            return index
        except Exception as e:
            print(f"⚠️ Failed to load existing index ({e}), rebuilding...")

    # If no saved index or loading failed, rebuild
    return build_index()
