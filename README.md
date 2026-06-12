# Synthetic Data for Entity Linking Domain Adaptation

## 1. Zero-Shot MCQ Benchmark on ELGOLD

Guide pour reproduire le benchmark zero-shot avec génération de candidats sur ELGOLD.

### Prérequis

```bash
pip install -r requirements.txt
```

Packages clés : `transformers`, `torch`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `wikimapper`, `pandas`, `tqdm`

---

### Étape 1 : Construire la base d'entités Wikipedia

Télécharger le dump Wikipedia et construire la base SQLite wikimapper (une seule fois, ~35 min) :

```bash
python utils/extract_all_entities.py --setup --wikimapper-dir data/wikimapper
```

Extraire tous les titres Wikipedia + QIDs Wikidata dans un CSV :

```bash
python utils/extract_all_entities.py --db data/wikimapper/index_enwiki-latest.db --out data/wikidata_qid_titles.csv
```

Vérifier : le fichier `data/wikidata_qid_titles.csv` contient deux colonnes (`qid`, `title`), dédupliqué par QID (~6-7M lignes).

---

### Étape 2 : Construire l'index de candidats

**BM25 (sparse retrieval)** — quelques secondes sur CPU :

```bash
python candidates/bm25_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/bm25.pkl
```

**Dense retrieval** — nécessite GPU, plusieurs heures :

```bash
python candidates/dense_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/
```

Les index sont sauvegardés sur disque et chargés instantanément par la suite.

---

### Étape 3 : Lancer l'évaluation

**MCQ + BM25 :**

```bash
python evaluation/eval_accuracy.py --mode mcq_bm25 --model Qwen/Qwen3-4B
```

**MCQ + Dense :**

```bash
python evaluation/eval_accuracy.py --mode mcq_dense --model Qwen/Qwen3-4B
```

Options :

| Flag | Description |
|---|---|
| `--n-samples 3` | Nombre de samples pour self-consistency (défaut: 3) |
| `--temperature 0.6` | Température de sampling (défaut: 0.6) |
| `--max-new-tokens 2048` | Tokens max pour le reasoning (défaut: 2048) |
| `--max-examples 200` | Limiter le nombre d'exemples (pour tester) |
| `--model Qwen/Qwen3-1.7B` | Changer de modèle |

Les résultats sont sauvegardés dans :
- `evaluation/results/{mode}_{model}.csv` — prédictions détaillées
- `evaluation/results/{mode}_{model}_metrics.json` — métriques agrégées

---

### Étape 4 : Analyser les résultats

```bash
jupyter notebook notebooks/zero_shot_benchmark.ipynb
```

Le notebook permet de visualiser les résultats, inspecter les erreurs, et comparer les configurations (BM25 vs Dense vs Fusion).

---

## Structure du projet

```
├── candidates/
│   ├── __init__.py
│   ├── bm25_retriever.py        # Retrieval sparse (BM25) — build + query
│   ├── dense_retriever.py       # Retrieval dense (Qwen3-Embedding + FAISS) — build + query
│   └── index/                   # Index sérialisés (bm25.pkl, dense.faiss) — gitignored
│
├── data/
│   ├── benchmarks/              # Fichiers ELGOLD (.txt annotés)
│   ├── wikimapper/              # Dump Wikipedia + base SQLite — gitignored
│   └── wikidata_qid_titles.csv  # Tous les titres Wikipedia + QIDs — gitignored
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_accuracy.py         # Script principal d'évaluation (cluster)
│   └── results/                 # CSV + JSON des résultats — gitignored
│
├── notebooks/
│   └── zero_shot_benchmark.ipynb
│
└── utils/
    ├── __init__.py
    ├── elgold_parser.py         # Parsing du benchmark ELGOLD
    ├── extract_all_entities.py  # Extraction des entités via wikimapper
    ├── llm.py                   # Chargement LLM, génération, self-consistency
    ├── metrics.py               # Accuracy, Recall@K, normalisation
    ├── prompt_builder.py        # Construction du prompt MCQ (indices numériques)
    ├── response_parser.py       # Parsing de la réponse LLM (index → titre)
    └── wiki_descriptions.py     # Fetch + cache des descriptions Wikipedia
```

---

## Référence des scripts

### `utils/extract_all_entities.py`

Extrait tous les titres Wikipedia + QIDs Wikidata depuis la base wikimapper. Déduplique par QID (garde le titre canonique le plus court).

```
--setup                 Télécharge le dump + construit la base SQLite
--wikimapper-dir PATH   Dossier pour le dump + DB (défaut: data/wikimapper)
--db PATH               Chemin vers la base SQLite wikimapper
--out PATH              CSV de sortie (défaut: data/wikidata_qid_titles.csv)
```

### `candidates/bm25_retriever.py`

Index BM25 sparse sur les titres Wikipedia. Détecte automatiquement le format CSV (colonne `title`) ou texte brut.

```
--titles PATH           CSV d'entrée (colonnes qid, title)
--out PATH              Fichier .pkl de sortie (défaut: candidates/index/bm25.pkl)
--max-entities N        Limiter le nombre d'entités (pour tester)
```

### `candidates/dense_retriever.py`

Index dense Qwen3-Embedding-4B + FAISS. Utilise `sentence-transformers` pour le pooling asymétrique (query vs document).

```
--titles PATH           CSV d'entrée (colonnes qid, title)
--out PATH              Dossier de sortie (défaut: candidates/index/)
--model NAME            Modèle d'embedding (défaut: Qwen/Qwen3-Embedding-4B)
--batch-size N          Taille du batch d'encodage (défaut: 256)
--max-entities N        Limiter le nombre d'entités (pour tester)
```

### `evaluation/eval_accuracy.py`

Évaluation complète : retrieval → descriptions → prompt MCQ → LLM (avec self-consistency) → parsing → métriques.

```
--mode MODE             mcq_bm25 | mcq_dense | zero_shot
--model NAME            Modèle HuggingFace (ex: Qwen/Qwen3-4B)
--n-samples N           Samples de self-consistency (défaut: 3)
--temperature F         Température de sampling (défaut: 0.6)
--max-new-tokens N      Tokens max pour reasoning (défaut: 2048)
--max-examples N        Limiter le benchmark (pour tester)
```

### `utils/llm.py`

Fonctions LLM partagées entre le notebook et le script d'évaluation.

- `load_model(name)` — charge le modèle + tokenizer
- `generate_batch(prompt, tokenizer, model, ...)` — génère N samples pour un prompt
- `majority_vote(raw_outputs, candidates)` — vote majoritaire sur les prédictions

### `utils/elgold_parser.py`

Parsing du benchmark ELGOLD.

- `parse_elgold_file(path)` — extrait les mentions annotées d'un fichier `.txt`
- `load_benchmark(dir, max_examples)` — charge tous les fichiers et retourne un DataFrame

### `utils/prompt_builder.py`

Construit le prompt MCQ : contexte (mention entre crochets) + type NER + candidats numérotés `0. Titre - Description`. Le LLM répond `"answer: 3"`.

### `utils/response_parser.py`

Parse la réponse du LLM : cherche le pattern `answer: N`, puis un index en début de ligne, puis un index isolé. Gère les blocs `<think>` et les guillemets.

### `utils/wiki_descriptions.py`

Fetch les descriptions Wikipedia via l'API REST. Cache sur disque (`data/wiki_descriptions_cache.json`) pour éviter les appels répétés.

### `utils/metrics.py`

M�triques : `normalize_entity()`, `is_correct()`, `recall_at_k()`, `compute_metrics()`, `print_metrics()`.