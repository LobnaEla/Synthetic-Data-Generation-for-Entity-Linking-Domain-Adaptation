"""Shared ELGOLD domain/category naming.

ELGOLD files are named `<category_id>_<doc_num>.txt` where category_id is one
of {1, 2, 3, 4, 5, 6a..6e, 8}. Subcategories of category 6 (Scientific
abstracts) are aggregated under a single domain, following the elgold-toolset
convention (`parsed_file['category'][0]` -> "ignore subcategories").

This mapping is the single source of truth for domain names, shared by the
benchmark loader (`utils/elgold_parser.py`), the evaluation script
(`evaluation/eval_accuracy.py`), and the entity-extraction analysis script
(`scripts/extract_elgold_entities.py`).
"""

CATEGORY_NAMES = {
    "1": "News",
    "2": "Job_offers",
    "3": "Movie_reviews",
    "4": "Automotive_blogs",
    "5": "Amazon_product_reviews",
    "6": "Scientific_abstracts",
    "8": "Historic_blogs",
}


def category_from_filename(name: str) -> str:
    """`6a_3.txt` -> 'Scientific_abstracts'. Aggregates sub-categories (6a..6e -> 6)."""
    prefix = name.split("_", 1)[0]
    numeric = "".join(c for c in prefix if c.isdigit())
    return CATEGORY_NAMES.get(numeric, f"category_{numeric or prefix}")
