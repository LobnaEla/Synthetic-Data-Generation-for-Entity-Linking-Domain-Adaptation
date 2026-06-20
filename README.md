# Synthetic Data for Entity Linking Domain Adaptation

## 1. Zero-Shot MCQ Benchmark on ELGOLD

Guide to reproduce the zero-shot benchmark with candidate generation on ELGOLD.

### Prerequisites

```bash
pip install -r requirements.txt
```

Key packages: `transformers`, `torch`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `wikimapper`, `pandas`, `tqdm`

---

### Step 1: Build the Wikipedia entity base

First run (downloads the dump, builds the DB, extracts entities, ~35 min):

```bash
python utils/extract_all_entities.py --setup --wikimapper-dir data/wikimapper
```

Re-run only the CSV extraction (DB already built):

```bash
python utils/extract_all_entities.py --db data/wikimapper/index_enwiki-latest.db
```

Result: `data/wikidata_qid_titles.csv` — two columns (`qid`, `title`), ~6-7M rows, deduplicated by QID.

---

### Step 2: Build the candidate indexes

**BM25 (sparse retrieval)** — a few seconds on CPU:

```bash
python candidates/bm25_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/bm25.pkl
```

**Dense retrieval** — requires a GPU, about 34 hours:

```bash
python candidates/dense_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/
```

Indexes are saved to disk and loaded instantly afterwards.

---

### Step 3: Pre-compute candidates

Pre-computing candidates for all ELGOLD mentions avoids reloading the retriever on every evaluation run.
The evaluation script auto-detects pre-computed files present in `candidates/precomputed/`.

```bash
# BM25 candidates
python candidates/precompute_candidates.py --retriever bm25 --k 100

# Dense candidates
python candidates/precompute_candidates.py --retriever dense --k 64
```

Expected output:

```text
candidates/precomputed/bm25_k100.json
candidates/precomputed/dense_k64.json
```

Notes:

* `bm25_k100.json` contains the top-100 BM25 candidates per mention.
* `dense_k64.json` contains the top-64 Dense candidates per mention.
* If these files exist, `evaluation/eval_accuracy.py` loads them automatically.
* If they don't exist, the script falls back to live retrieval from the indexes in `candidates/index/`.

---

### Step 4: Run the evaluation

The evaluation pipeline is:

```text
Retriever candidates → optional Reranker → MCQ Prompt → LLM → Parsing → Metrics
```

By default, the reranker is enabled, and evaluation is run **separately for each ELGOLD domain**
(News, Job_offers, Movie_reviews, Automotive_blogs, Amazon_product_reviews, Scientific_abstracts,
Historic_blogs — see [Domain split](#domain-split) below).

#### BM25 + Reranker + LLM

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --batch-size 8
```

With this configuration:

```text
BM25 top-100 → Qwen3-Reranker top-10 → Qwen3-4B LLM
```

The reranker defaults to:

```text
Qwen/Qwen3-Reranker-4B
```

The number of candidates kept after reranking is controlled by:

```bash
--k 10
```

Example:

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --k 10 \
    --batch-size 8
```

#### Dense + Reranker + LLM

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_dense \
    --model Qwen/Qwen3-4B \
    --k 10 \
    --batch-size 8
```

With this configuration:

```text
Dense top-64 → Qwen3-Reranker top-10 → Qwen3-4B LLM
```

#### Fusion BM25 + Dense + Reranker + LLM

```bash
python evaluation/eval_accuracy.py \
    --mode fusion \
    --model Qwen/Qwen3-4B \
    --k 10 \
    --batch-size 8
```

With this configuration:

```text
BM25 top-100 + Dense top-64 → union candidates → Qwen3-Reranker top-10 → Qwen3-4B LLM
```

Fusion merges BM25 and Dense candidates, removing duplicates and keeping BM25 order
first, then Dense additions.

---

### Evaluation without a reranker

To disable the reranker:

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --no-reranker \
    --batch-size 8
```

Important:

* With reranker: `--k` limits the number of candidates kept after reranking.
* Without reranker: all retrieved candidates are passed to the LLM, with no top-k truncation.
* So in `--no-reranker` mode, `--k` is ignored for candidate selection.

Example:

```text
BM25 with --no-reranker:
BM25 top-100 → Qwen3-4B LLM

Dense with --no-reranker:
Dense top-64 → Qwen3-4B LLM

Fusion with --no-reranker:
BM25 top-100 + Dense top-64 → union candidates → Qwen3-4B LLM
```

---

### Running stages separately: `--stage`

Reranking (CPU/GPU-light cross-encoder) and the final LLM call have very different
resource profiles, and on a single GPU they must not be loaded at the same time
(see `candidates/reranker.py`). `--stage` lets you split the pipeline into
independent runs:

| Stage           | What it does                                                                          |
| --------------- | -------------------------------------------------------------------------------------- |
| `full` (default)| Retrieval → reranking → LLM evaluation, in one run. Reranked candidates are also saved to `candidates/reranked/` for later reuse. |
| `rerank`         | Retrieval → reranking only. Saves reranked candidates to `candidates/reranked/` and exits — **no LLM is loaded**. |
| `llm`            | Skips retrieval/reranking entirely. Loads pre-computed reranked candidates from `candidates/reranked/` and runs only the LLM evaluation. |

This lets you, for example, run reranking on one job/GPU slot and the LLM
evaluation on another, without reloading both models together:

```bash
# 1. Rerank only, save reranked candidates to disk
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --stage rerank \
    --k 10

# 2. LLM evaluation only, using the reranked candidates saved above
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --stage llm \
    --model Qwen/Qwen3-4B \
    --k 10 \
    --batch-size 8
```

Reranked candidates are saved as one JSON file per `(mode, domain, k)` combination:

```text
candidates/reranked/{mode}_{domain}_k{k}.json
```

`--stage llm` requires this file to already exist for every domain being evaluated
(run `--stage rerank` or `--stage full` first). `--model` is required for
`--stage full` and `--stage llm`, but not for `--stage rerank`.

---

### Domain split: `--domains`

ELGOLD covers 7 domains: `News`, `Job_offers`, `Movie_reviews`, `Automotive_blogs`,
`Amazon_product_reviews`, `Scientific_abstracts`, `Historic_blogs`. The domain of
each file is derived directly from its filename (`utils/elgold_categories.py`,
e.g. `1_3.txt` → `News`, `6b_2.txt` → `Scientific_abstracts`).

By default (`--domains all`), the evaluation loop runs **separately for every
domain**: each domain gets its own retrieval/recall numbers, its own reranked
candidates file, its own results CSV, and its own metrics JSON. A combined
`*_domains_summary.json` is saved at the end with per-domain accuracy.

To restrict evaluation to a subset of domains:

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --domains News,Job_offers
```

> Note: this domain split only requires the filename-based mapping in
> `utils/elgold_categories.py` — it does not depend on the
> `scripts/extract_elgold_entities.py` analysis tool below.

---

### Using pre-computed retrieval candidates

No specific flag is needed.

The script automatically looks for:

```text
candidates/precomputed/bm25_k*.json
candidates/precomputed/dense_k*.json
```

Depending on the selected mode:

| Mode              | Pre-computed file used                 |
| ----------------- | --------------------------------------- |
| `zero_shot_bm25`  | `candidates/precomputed/bm25_k*.json`  |
| `zero_shot_dense` | `candidates/precomputed/dense_k*.json` |
| `fusion`          | Both the BM25 and Dense files           |

If several files match the same pattern, the last one in sorted order is used.

---

### Useful options

| Flag                                | Description                                                          |
| ------------------------------------ | --------------------------------------------------------------------- |
| `--mode zero_shot_bm25`             | Evaluation with BM25 candidates                                       |
| `--mode zero_shot_dense`            | Evaluation with Dense candidates                                      |
| `--mode fusion`                     | Fusion of BM25 and Dense candidates                                   |
| `--stage full`                      | Retrieval + reranking + LLM in one run (default)                      |
| `--stage rerank`                    | Retrieval + reranking only, saves reranked candidates, no LLM         |
| `--stage llm`                       | LLM evaluation only, using pre-computed reranked candidates           |
| `--domains all`                     | Evaluate every ELGOLD domain separately (default)                     |
| `--domains News,Job_offers`         | Evaluate only the listed domains                                      |
| `--model Qwen/Qwen3-4B`             | LLM used to pick the final answer (required unless `--stage rerank`)  |
| `--reranker Qwen/Qwen3-Reranker-4B` | Reranker model, used unless `--no-reranker`                           |
| `--no-reranker`                     | Disables the reranker and passes all candidates to the LLM            |
| `--k 10`                            | Number of candidates kept after reranking                             |
| `--batch-size 8`                    | LLM batch size                                                        |
| `--max-new-tokens 2048`             | Maximum number of generated tokens                                    |
| `--max-examples 200`                | Limit the benchmark size for quick testing                            |
| `--reranked-dir PATH`               | Where reranked candidates are saved/read (default: `candidates/reranked/`) |
| `--resume`                          | Resume from checkpoint                                                |

---

### Results

Results are saved per domain in:

```text
evaluation/results/{mode}_{domain}_{model}_{timestamp}.csv
evaluation/results/{mode}_{domain}_{model}_{timestamp}_metrics.json
evaluation/results/{mode}_{model}_{timestamp}_domains_summary.json
```

The CSV file notably contains:

| Column                  | Description                                                          |
| ------------------------ | ---------------------------------------------------------------------|
| `mention`               | ELGOLD mention                                                        |
| `domain`                | ELGOLD domain the mention's file belongs to                           |
| `gold`                  | Correct entity                                                        |
| `candidates_retrieval`  | Candidates retrieved by BM25/Dense/Fusion                             |
| `candidates_reranked`   | Candidates after reranking, or all candidates if `--no-reranker`      |
| `gold_in_retrieval`     | Whether the gold entity is in the retrieved candidates                |
| `gold_in_reranked`      | Whether the gold entity is in the candidates passed to the LLM        |
| `predicted`             | Entity predicted by the LLM                                           |
| `correct`               | Whether the prediction is correct                                     |
| `raw_output`            | Raw LLM output                                                        |
| `error`                 | Any error or unparsable output                                        |

---

### Resuming after an interruption

If a job is interrupted, a checkpoint is automatically saved in `evaluation/results/`
(one checkpoint per domain). Resume with:

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --resume
```

The script resumes already-processed examples from the checkpoint.

### Step 5: Analyze results

```bash
jupyter notebook notebooks/zero_shot_benchmark.ipynb
```

The notebook lets you visualize results, inspect errors, and compare
configurations (BM25 vs Dense vs Fusion) and domains.

---

## 2. ELGOLD domain entity statistics (optional analysis tool)

`scripts/extract_elgold_entities.py` is a standalone analysis tool that summarizes,
for each ELGOLD domain, how many gold entities/mentions/NIL mentions appear, and
maps each gold title to its Wikidata QID via `wikimapper`. It is **not** required
to run the per-domain evaluation above — domain detection there only needs the
filename-based mapping in `utils/elgold_categories.py`. Use this script when you
need entity-level coverage statistics per domain (e.g. to scope a domain-specific
candidate index, or to report dataset statistics).

```bash
python scripts/extract_elgold_entities.py data/benchmarks \
    data/elgold_entities_per_domain.json \
    --wikimapper_db data/wikimapper/index_enwiki-latest.db
```

Output: `data/elgold_entities_per_domain.json` (gitignored, like the rest of `data/`),
with one entry per domain: `n_docs`, `n_mentions`, `n_nil`, `n_unique_titles`,
`n_mapped_qids`, `n_unmapped_titles`, `entities_qids`, `title_to_qid`, `unmapped_titles`.

---

## Project structure

```
├── candidates/
│   ├── __init__.py
│   ├── bm25_retriever.py           # BM25 sparse retrieval
│   ├── dense_retriever.py          # Dense retrieval with Qwen3-Embedding + FAISS
│   ├── reranker.py                 # Candidate reranking with Qwen3-Reranker
│   ├── precompute_candidates.py    # Pre-computes BM25/Dense candidates
│   ├── index/                      # Serialized indexes (.pkl, .faiss) — gitignored
│   ├── precomputed/                # Pre-computed retrieval candidates (JSON)
│   └── reranked/                   # Pre-computed reranked candidates (JSON)
│
├── data/
│   ├── benchmarks/                 # ELGOLD files (annotated .txt) — gitignored
│   ├── wikimapper/                 # Wikipedia dump + SQLite DB — gitignored
│   ├── wikidata_qid_titles.csv     # All Wikipedia titles + QIDs — gitignored
│   └── elgold_entities_per_domain.json  # Per-domain entity stats — gitignored
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_accuracy.py            # Main evaluation script (cluster)
│   └── results/                    # Results CSV + JSON — gitignored
│
├── notebooks/
│   └── zero_shot_benchmark.ipynb
│
├── scripts/
│   └── extract_elgold_entities.py  # Per-domain entity stats (analysis tool)
│
├── slurm/                          # SLURM scripts for the cluster
│   ├── run_eval_bm25.sh
│   ├── run_eval_dense.sh
│   └── run_precompute_bm25.sh
│
└── utils/
    ├── __init__.py
    ├── elgold_categories.py        # ELGOLD filename → domain mapping (shared)
    ├── elgold_parser.py            # ELGOLD benchmark parsing
    ├── extract_all_entities.py     # Entity extraction via wikimapper
    ├── llm.py                      # LLM loading + batch generation
    ├── metrics.py                  # Accuracy, Recall@K, normalization
    ├── prompt_builder.py           # MCQ prompt construction (numbered choices)
    └── response_parser.py          # LLM response parsing (index → title)
```

---

## Script reference

### `utils/extract_all_entities.py`

Extracts all Wikipedia titles + Wikidata QIDs from the wikimapper database. Deduplicates by QID (keeps the canonical title).

```
--setup                 Downloads the dump + builds the SQLite DB
--wikimapper-dir PATH   Directory for the dump + DB (default: data/wikimapper)
--db PATH               Path to the wikimapper SQLite DB
--out PATH              Output CSV (default: data/wikidata_qid_titles.csv)
```

### `candidates/bm25_retriever.py`

Sparse BM25 index over Wikipedia titles. Auto-detects CSV format (`title` column) or plain text.

```
--titles PATH           Input CSV (qid, title columns)
--out PATH              Output .pkl file (default: candidates/index/bm25.pkl)
--max-entities N        Limit the number of entities (for testing)
```

### `candidates/dense_retriever.py`

Dense Qwen3-Embedding-4B + FAISS index. Uses `sentence-transformers` for asymmetric pooling (query vs document).

```
--titles PATH           Input CSV (qid, title columns)
--out PATH              Output directory (default: candidates/index/)
--model NAME            Embedding model (default: Qwen/Qwen3-Embedding-4B)
--batch-size N          Encoding batch size (default: 256)
--max-entities N        Limit the number of entities (for testing)
```

### `candidates/precompute_candidates.py`

Pre-computes candidates for every mention in the ELGOLD benchmark and saves them to JSON.

```
--retriever TYPE        bm25 | dense
--k N                   Number of candidates per mention (default: 100)
--bm25-index PATH       Path to the BM25 index
--dense-index PATH      Path to the dense index
--out PATH              Output JSON file (auto-named if omitted)
```

### `evaluation/eval_accuracy.py`

Main zero-shot evaluation script on ELGOLD.

Full pipeline:

```text
candidate retrieval → optional reranking → MCQ prompt → LLM generation → parsing → metrics
```

Run separately per ELGOLD domain (`--domains`), and splittable into independent
stages (`--stage full|rerank|llm`) so the reranker and the final LLM never have
to share GPU memory.

Available modes:

```text
zero_shot_bm25
zero_shot_dense
fusion
```

Main arguments:

```text
--mode MODE             zero_shot_bm25 | zero_shot_dense | fusion
--stage STAGE           full | rerank | llm (default: full)
--domains LIST          Comma-separated domain names, or 'all' (default: all)
--model NAME            HuggingFace model used as the final LLM
--reranker NAME         HuggingFace model used as the reranker
--no-reranker           Disables the reranker
--k N                   Number of candidates kept after reranking
--batch-size N          LLM batch size
--max-new-tokens N      Maximum number of generated tokens
--max-examples N        Limit the benchmark size for testing
--index PATH            Path to the BM25 index
--dense-index PATH      Path to the Dense index
--benchmarks PATH       Path to the ELGOLD files
--results-dir PATH      Directory where results are saved
--reranked-dir PATH     Directory where reranked candidates are saved/read
--resume                Resume from checkpoint
```

How pre-computed retrieval candidates work:

* The script automatically looks for files in `candidates/precomputed/`.
* In BM25 mode, it looks for `bm25_k*.json`.
* In Dense mode, it looks for `dense_k*.json`.
* In Fusion mode, it looks for both.
* If no pre-computed file is found, the retriever is loaded and candidates are computed live.

How the reranker works:

* The reranker is enabled by default.
* It receives the candidates retrieved by BM25, Dense, or Fusion.
* It reorders the candidates and keeps the top-`k`.
* The LLM only receives the reranked candidates.
* With `--no-reranker`, the LLM receives all retrieved candidates, untruncated.
* With `--stage full` or `--stage rerank`, reranked candidates are saved to `candidates/reranked/{mode}_{domain}_k{k}.json` for later reuse with `--stage llm`.

Examples:

```bash
# BM25 + reranker, full pipeline, all domains
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --k 10

# Rerank only (no LLM), single domain
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --stage rerank \
    --domains News \
    --k 10

# LLM only, using the reranked candidates from the run above
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --stage llm \
    --domains News \
    --model Qwen/Qwen3-4B \
    --k 10

# Fusion + reranker
python evaluation/eval_accuracy.py \
    --mode fusion \
    --model Qwen/Qwen3-4B \
    --k 10

# BM25 without reranker
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --no-reranker
```

### `scripts/extract_elgold_entities.py`

Standalone analysis tool: per-domain gold entity/mention/NIL statistics, with
titles mapped to Wikidata QIDs via wikimapper. Not required for the per-domain
evaluation split (see [ELGOLD domain entity statistics](#2-elgold-domain-entity-statistics-optional-analysis-tool) above).

```
data_root               Path to ELGOLD data (flat .txt files)
output                  Output JSON summary file
--wikimapper_db PATH    Path to the wikimapper SQLite DB
--glob PATTERN          Glob for ELGOLD files (default: *.txt)
```

### `utils/llm.py`

LLM functions shared between the notebook and the evaluation script.

- `load_model(name)` — loads the model + tokenizer
- `generate_batch(prompts, tokenizer, model, max_new_tokens)` — generates one response per prompt, in batch

### `utils/elgold_categories.py`

Single source of truth for the ELGOLD filename → domain mapping, shared by
`elgold_parser.py`, `eval_accuracy.py`, and `extract_elgold_entities.py`.

- `CATEGORY_NAMES` — category id → domain name
- `category_from_filename(name)` — `"6b_2.txt"` → `"Scientific_abstracts"`

### `utils/elgold_parser.py`

ELGOLD benchmark parsing.

- `parse_elgold_file(path)` — extracts annotated mentions from a `.txt` file, tagged with their domain
- `load_benchmark(dir, max_examples)` — loads all files and returns a DataFrame (with a `domain` column)

### `utils/prompt_builder.py`

Builds the MCQ prompt: context (mention in square brackets) + NER type + numbered candidates `0. Title`. The LLM answers `"answer: 3"`.

### `utils/response_parser.py`

Parses the LLM response: looks for the `answer: N` pattern, then an index at the start of a line, then a standalone index. Handles `<think>` blocks and quotes.

### `utils/metrics.py`

Metrics: `normalize_entity()`, `is_correct()`, `recall_at_k()`, `compute_metrics()`, `print_metrics()`.
