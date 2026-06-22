# Synthetic Data for Entity Linking Domain Adaptation

## Contribution Section

**Soline Carle** Wrote the project plan, the project report and the project presentation slides. She led the Wikidata subgraph extraction.

**Lobna Elabed** Made the zero shot, designed and implemented the candidate generation module.

**Anas Elbarhdadi** Led the human-data fine-tuning branch (branch 2) in its entirety.

**Luciano Nichelatti** and **Nour Ghamgui** jointly developed the synthetic data generation pipeline (branch 3).

**Team organization:** the team used a shared GitLab repository on the Télécom Paris infrastructure, with one branch per team member. Weekly meetings were held with supervisor Samy Haffoudhi to review progress and adjust priorities. Between meetings, the team synchronized via a WhatsApp group. The project plan (`Project_plan.pdf`), written at the start of the project, defined task ownership, deadlines, and risk mitigation strategies.


---------------------------------------------------------------------------------------


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