# Synthetic Data for Entity Linking Domain Adaptation

**fetch_wikidata_titles.py**
Fetch all English Wikipedia titles from Wikidata via paginated SPARQL.

Strategy: iterate over QID ranges (Q1–Q1_000_000, Q1_000_001–Q2_000_000, ...) instead of OFFSET pagination — Wikidata times out on large OFFSETs.

Output: data/kg/wikipedia_titles.txt  (one title per line, UTF-8)

Resume-safe: already-fetched ranges are skipped.

Usage:
    python fetch_wikidata_titles.py
    python fetch_wikidata_titles.py --max-qid 2000000   # partial fetch for testing
    python fetch_wikidata_titles.py --out custom_path.txt


**utils/**
**prompt_builder.py**
Prompt builder for entity linking.

Two modes:
  - zero_shot : no candidates, LLM directly predicts the Wikipedia title.
  - mcq       : candidates list provided, LLM picks A/B/C/...

**response_parser.py**
Response parser for entity linking LLM outputs.

Two modes:
  - zero_shot : extract first non-empty line as the predicted title
  - mcq       : extract a letter (A/B/C...) and map it to the candidate list

**metrics.py**
Evaluation metrics for entity linking.

Functions:
  - normalize_entity(s): lowercase + strip + underscores → spaces
  - is_correct(pred, gold): normalized exact match
  - recall_at_k(gold, candidates): is gold in top-k candidates?
  - compute_metrics(df): aggregate accuracy + recall@k from a results DataFrame


**candidates/**
**bm25_retriever.py**
BM25 sparse retriever for entity linking candidate generation.
Builds the BM25 index from wikipedia_titles.txt and saves it to disk.

Usage:
    retriever = BM25Retriever.from_entity_list(entity_titles)
    retriever.save("candidates/index/bm25.pkl")

    retriever = BM25Retriever.load("candidates/index/bm25.pkl")
    candidates = retriever.retrieve("Toronto Film School", k=10)

**evaluation/**
**eval_accuracy.py**
Script de benchmark ELGOLD.

Usage:
    python eval_accuracy.py --mode zero_shot --model Qwen/Qwen3-0.6B
    python eval_accuracy.py --mode mcq_bm25  --model Qwen/Qwen3-4B --batch-size 16
    python eval_accuracy.py --mode mcq_bm25  --model Qwen/Qwen3-4B --max-examples 200

Sorties:
    evaluation/results/{mode}_{model}.csv
    evaluation/results/{mode}_{model}_metrics.json