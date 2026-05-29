import argparse
import pickle
import sys
import time
from pathlib import Path
import pandas as pd
import torch 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        self.model_name = model_name
        self.entity_titles: list[str] = []
        self.index: faiss.Index | None = None
        self._model: SentenceTransformer | None = None

    def _load_encoder(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[Dense] Loading encoder: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
        return self._model

    def _encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        model = self._load_encoder()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        )
        return np.array(embeddings, dtype=np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        model = self._load_encoder()
        emb = model.encode(
            [query],
            prompt_name="query",
            normalize_embeddings=True,
        )
        return np.array(emb, dtype=np.float32)

    def build_index(self, entity_titles: list[str], batch_size: int = 256) -> None:
        print(f"[Dense] Encoding {len(entity_titles):,} entities ...")
        self.entity_titles = entity_titles
        embeddings = self._encode(entity_titles, batch_size=batch_size)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"[Dense] FAISS index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "dense.faiss"))
        with open(index_dir / "dense_meta.pkl", "wb") as f:
            pickle.dump({"titles": self.entity_titles, "model_name": self.model_name}, f)
        print(f"[Dense] Saved to {index_dir} ({len(self.entity_titles):,} entities)")

    @classmethod
    def load(cls, index_dir: str | Path) -> "DenseRetriever":
        index_dir = Path(index_dir)
        with open(index_dir / "dense_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(model_name=meta["model_name"])
        obj.entity_titles = meta["titles"]
        obj.index = faiss.read_index(str(index_dir / "dense.faiss"))
        print(f"[Dense] Loaded: {obj.index.ntotal} vectors")
        return obj

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        emb = self._encode_query(query)
        _, indices = self.index.search(emb, k)
        return [self.entity_titles[i] for i in indices[0] if i >= 0]

    def retrieve_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        emb = self._encode_query(query)
        scores, indices = self.index.search(emb, k)
        return [
            (self.entity_titles[i], float(scores[0][j]))
            for j, i in enumerate(indices[0])
            if i >= 0
        ]
    
def main():
    parser = argparse.ArgumentParser(description="Build dense FAISS index from Wikipedia titles")
    parser.add_argument("--titles", required=True, help="Path to CSV")
    parser.add_argument("--out", default="candidates/index/", help="Output directory")
    parser.add_argument("--title-col", default="title", help="Column name in CSV")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-entities", type=int, default=None, help="Limit for testing")
    args = parser.parse_args()
  
    titles_path = Path(args.titles)
    if not titles_path.exists():
        print(f"[ERROR] File not found: {titles_path}")
        sys.exit(1)
 
    print(f"Loading titles from {titles_path} ...")
    df = pd.read_csv(titles_path)
    if args.title_col not in df.columns:
        print(f"[ERROR] Column '{args.title_col}' not found. Available: {df.columns.tolist()}")
        sys.exit(1)
 
    titles = df[args.title_col].dropna().tolist()
    if args.max_entities:
        titles = titles[: args.max_entities]
    print(f"{len(titles):,} titles loaded")
 
    retriever = DenseRetriever(model_name=args.model)
 
    t0 = time.time()
    retriever.build_index(titles, batch_size=args.batch_size)
    print(f"Index built in {time.time() - t0:.1f}s")
 
    retriever.save(args.out)
 
    # Sanity check on a few examples
    print("\nTest on examples:")
    test_queries = [
        ("Robbie Coltrane", "Robbie Coltrane"),
        ("Scottish", "Scottish people"),
        ("Harry Potter", "Harry Potter (film series)"),
        ("British", "British people"),
        ("Cracker", "Cracker (British TV series)"),
    ]
    for mention, gold in test_queries:
        candidates = retriever.retrieve(mention, k=10)
        found = "✓" if gold.lower() in [c.lower() for c in candidates] else "✗"
        print(f"{found}  '{mention}'  →  gold='{gold}'")
        print(f"     {candidates}")
 
 
if __name__ == "__main__":
    main()