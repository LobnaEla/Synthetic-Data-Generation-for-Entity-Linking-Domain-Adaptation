import argparse
import time
from pathlib import Path
import pickle
import re, sys
<<<<<<< HEAD

=======
import csv
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, entity_titles: list[str]):
        self.entity_titles = entity_titles
        tokenized = [_tokenize(t) for t in entity_titles]
        self.bm25 = BM25Okapi(tokenized)

    @classmethod
    def from_entity_list(cls, entity_titles: list[str]) -> "BM25Retriever":
        return cls(entity_titles)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"titles": self.entity_titles, "bm25": self.bm25}, f)
        print(f"[BM25] Index saved to {path} ({len(self.entity_titles)} entities)")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.entity_titles = data["titles"]
        obj.bm25 = data["bm25"]
        print(f"[BM25] Index loaded from {path} ({len(obj.entity_titles)} entities)")
        return obj

    def retrieve(self, query: str, k: int = 10) -> list[str]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.entity_titles[i] for i in top_indices]

    def retrieve_with_scores(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.entity_titles[i], float(scores[i])) for i in top_indices]
    


def load_titles(titles_path: Path, max_entities: int | None = None) -> list[str]:
    print(f"Loading titles from {titles_path} ...")
    titles = []
    with open(titles_path, encoding="utf-8") as f:
<<<<<<< HEAD
        for i, line in enumerate(f):
            if max_entities and i >= max_entities:
                break
            t = line.strip()
            if t:
                titles.append(t)
=======
        first_line = f.readline().strip()
        f.seek(0)

        if "," in first_line and ("qid" in first_line.lower() or "title" in first_line.lower()):
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_entities and i >= max_entities:
                    break
                t = row.get("title", "").strip()
                if t:
                    titles.append(t)
        else:
            for i, line in enumerate(f):
                if max_entities and i >= max_entities:
                    break
                t = line.strip()
                if t:
                    titles.append(t)

>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
    print(f"  {len(titles):,} titles loaded")
    return titles


def main():
    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument("--titles", default="data/kg/wikipedia_titles.txt")
=======
    parser.add_argument("--titles", default="data/wikidata_qid_titles.csv")
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
    parser.add_argument("--out", default="candidates/index/bm25.pkl")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="Limit number of entities (for testing)")
    args = parser.parse_args()

    titles_path = Path(args.titles)
    out_path = Path(args.out)

    if not titles_path.exists():
        print(f"[ERROR] Titles file not found: {titles_path}")
<<<<<<< HEAD
        print("Run fetch_wikidata_titles.py first.")
=======
        print("Run extract_all_entities.py first.")
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
        sys.exit(1)

    titles = load_titles(titles_path, args.max_entities)

    print("Building BM25 index ...")
    t0 = time.time()
    retriever = BM25Retriever.from_entity_list(titles)
    elapsed = time.time() - t0
    print(f"  Built in {elapsed:.1f}s")

    retriever.save(out_path)

    # Test
    print("\nSanity checks:")
    for query in ["Toronto Film School", "Paris", "Barack Obama", "CRISPR"]:
        candidates = retriever.retrieve(query, k=5)
        print(f"  '{query}' → {candidates}")


if __name__ == "__main__":
    main()