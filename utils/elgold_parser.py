import glob
import re
from pathlib import Path

import pandas as pd

ANNOT_RE = re.compile(r"\{\{([^|{}]+)\|([^|{}]+)\|([^{}]+)\}\}")


def parse_elgold_file(path: Path, context_window: int = 150) -> list[dict]:
    raw = path.read_text(encoding="utf-8")
    clean = ANNOT_RE.sub(lambda m: m.group(1).strip(), raw)
    examples, cursor, last_end = [], 0, 0
    for match in ANNOT_RE.finditer(raw):
        mention = match.group(1).strip()
        ent_type = match.group(2).strip()
        gold = match.group(3).strip()
        cursor += len(raw[last_end : match.start()])
        end = cursor + len(mention)
        snippet = clean[max(0, cursor - context_window) : min(len(clean), end + context_window)]
        examples.append({
            "file": path.name, "mention": mention,
            "entity_type": ent_type, "gold": gold, "context": snippet,
        })
        cursor, last_end = end, match.end()
    return examples


def load_benchmark(benchmarks_dir: str, max_examples: int | None = None) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{benchmarks_dir}/*.txt"):
        rows.extend(parse_elgold_file(Path(f)))
    df = pd.DataFrame(rows).drop_duplicates()
    if max_examples:
        df = df.head(max_examples)
    return df