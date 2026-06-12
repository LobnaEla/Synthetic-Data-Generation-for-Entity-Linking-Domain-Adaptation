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
                model_kwargs={"dtype": torch.bfloat16},
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

    def build_index(
        self,
        entity_titles: list[str],
        batch_size: int = 256,
        index_chunk: int = 50_000,
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 5,
    ) -> None:
        self.entity_titles = entity_titles
        total = len(entity_titles)
        start_offset = 0

        # --- Resume from checkpoint if available ---
        ckpt_index_path = None
        ckpt_meta_path = None
        if checkpoint_dir is not None:
            ckpt_index_path = checkpoint_dir / "ckpt_dense.faiss"
            ckpt_meta_path = checkpoint_dir / "ckpt_dense_meta.pkl"

            if ckpt_index_path.exists() and ckpt_meta_path.exists():
                with open(ckpt_meta_path, "rb") as f:
                    ckpt_meta = pickle.load(f)
                start_offset = ckpt_meta["encoded_count"]
                self.index = faiss.read_index(str(ckpt_index_path))
                print(
                    f"[Dense] Resuming from checkpoint: {start_offset:,} / {total:,} "
                    f"already indexed ({self.index.ntotal} vectors)"
                )

        # --- Create fresh index if no checkpoint ---
        if self.index is None:
            print(f"[Dense] Encoding {total:,} entities from scratch ...")
            probe = self._encode(entity_titles[:2], batch_size=2)
            dim = probe.shape[1]
            self.index = faiss.IndexScalarQuantizer(
                dim, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(probe)
            print(f"[Dense] FAISS fp16 index initialised: dim={dim}")

        # --- Encode remaining chunks ---
        chunks_since_ckpt = 0
        t_start = time.time()
        chunks_done_this_run = 0

        for start in range(start_offset, total, index_chunk):
            t_chunk = time.time()
            chunk = entity_titles[start : start + index_chunk]
            embeddings = self._encode(chunk, batch_size=batch_size)
            self.index.add(embeddings)
            del embeddings

            encoded_so_far = min(start + index_chunk, total)
            chunks_done_this_run += 1
            elapsed = time.time() - t_start
            chunk_time = time.time() - t_chunk
            remaining_entities = total - encoded_so_far
            avg_per_entity = elapsed / (encoded_so_far - start_offset)
            eta_s = avg_per_entity * remaining_entities

            print(
                f"[Dense] Indexed {encoded_so_far:,} / {total:,}"
                f"  | chunk {chunk_time:.0f}s"
                f"  | elapsed {elapsed / 3600:.1f}h"
                f"  | ETA {eta_s / 3600:.1f}h"
            )

            # Periodic checkpoint
            chunks_since_ckpt += 1
            if (
                checkpoint_dir is not None
                and chunks_since_ckpt >= checkpoint_every
                and encoded_so_far < total
            ):
                self._save_checkpoint(checkpoint_dir, encoded_so_far)
                chunks_since_ckpt = 0

        total_elapsed = time.time() - t_start
        print(
            f"[Dense] Done: {self.index.ntotal} vectors"
            f"  | this run: {total_elapsed / 3600:.1f}h"
            f" ({chunks_done_this_run} chunks)"
        )

        # Clean up checkpoint files
        if ckpt_index_path is not None and ckpt_index_path.exists():
            ckpt_index_path.unlink()
            ckpt_meta_path.unlink()
            print("[Dense] Checkpoint files removed.")

    def _save_checkpoint(self, checkpoint_dir: Path, encoded_count: int) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(checkpoint_dir / "ckpt_dense.faiss"))
        with open(checkpoint_dir / "ckpt_dense_meta.pkl", "wb") as f:
            pickle.dump({
                "encoded_count": encoded_count,
                "titles": self.entity_titles,
                "model_name": self.model_name,
            }, f)
        print(f"[Dense] Checkpoint saved at {encoded_count:,} entities")

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
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save checkpoint every N chunks (default: 5)")
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
 
    out_dir = Path(args.out)
    retriever = DenseRetriever(model_name=args.model)
 
    t0 = time.time()
    retriever.build_index(
        titles,
        batch_size=args.batch_size,
        checkpoint_dir=out_dir,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Index built in {time.time() - t0:.1f}s")
 
    retriever.save(out_dir)
 
    # Sanity check
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