import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "shl_product_catalog.json")
EMBED_MODEL  = "all-MiniLM-L6-v2"

KEY_TO_CODE = {
    "Ability & Aptitude":              "A",
    "Assessment Exercises":            "E",
    "Biodata & Situational Judgment":  "B",
    "Biodata & Situational Judgement": "B",
    "Competencies":                    "C",
    "Development & 360":               "D",
    "Knowledge & Skills":              "K",
    "Personality & Behavior":          "P",
    "Simulations":                     "S",
}


def keys_to_test_type(keys: list[str]) -> str:
    """
    Converts a list of full key names to a comma-separated string of codes.
    Example: ["Knowledge & Skills", "Simulations"] -> "K,S"
    Unknown keys are kept as-is so nothing is silently dropped.
    """
    codes = [KEY_TO_CODE.get(key, key) for key in keys]
    return ",".join(codes)

#Load and prepare catalog 

def load_catalog(path: str) -> list[dict]:
    """
    Loads catalog JSON and adds two computed fields to each item:
        test_type  — short code string e.g. "K" or "A,P"
        text       — the string we embed for similarity search
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    catalog = []
    for item in raw:
        item["test_type"] = keys_to_test_type(item.get("keys", []))

        parts = [
            item.get("name", ""),
            item.get("name", ""),
            item.get("description", ""),
            " ".join(item.get("keys", [])),
            " ".join(item.get("job_levels", [])),
        ]
        item["text"] = " ".join(p for p in parts if p).strip()
        catalog.append(item)

    return catalog

#Build FAISS index 

def build_index(catalog: list[dict], model: SentenceTransformer):
    """
    Embeds all catalog items and stores them in a FAISS flat index.
    Uses cosine similarity (normalize first, then inner product).
    """
    texts   = [item["text"] for item in catalog]
    print(f"Embedding {len(texts)} catalog items...")
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(vectors)

    dim   = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    print(f"FAISS index ready: {index.ntotal} vectors, dim={dim}")
    return index


# Module-level startup (runs once when imported)

print("Loading embedding model...")
_model   = SentenceTransformer(EMBED_MODEL)

print("Loading catalog...")
_catalog = load_catalog(CATALOG_PATH)

_index = build_index(_catalog, _model)

#Public search function

def search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns the top_k most relevant catalog items for the given query.

    Args:
        query  — free text e.g. "hiring a mid-level Java developer"
        top_k  — how many results to return (we pass 10 to the LLM to pick from)

    Returns:
        List of catalog item dicts ordered by relevance, each containing:
        name, link, test_type, description, keys, job_levels, remote,
        adaptive, duration, languages
    """
    vec = _model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(vec)

    _distances, indices = _index.search(vec, top_k)

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(_catalog[idx])
    return results