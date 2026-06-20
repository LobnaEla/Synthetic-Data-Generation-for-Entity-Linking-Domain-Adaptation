# Synthetic Data for Entity Linking Domain Adaptation

<<<<<<< HEAD
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
    
**dense_retriever.py**
Dense retriever for entity linking candidate generation.
Uses Qwen3-Embedding-4B + FAISS.

Usage (as a module):
    retriever = DenseRetriever.load("candidates/index/")
    candidates = retriever.retrieve("Toronto Film School", k=10)
 
Usage (build index from CLI):
    python dense_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/
    python dense_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/ --max-entities 50000


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
=======
## 1. Zero-Shot MCQ Benchmark on ELGOLD

Guide pour reproduire le benchmark zero-shot avec génération de candidats sur ELGOLD.

### Prérequis

```bash
pip install -r requirements.txt
```

Packages clés : `transformers`, `torch`, `rank-bm25`, `sentence-transformers`, `faiss-cpu`, `wikimapper`, `pandas`, `tqdm`

---

### Étape 1 : Construire la base d'entités Wikipedia

Premier lancement (télécharge le dump, construit la DB, extrait les entités, ~35 min) :

```bash
python utils/extract_all_entities.py --setup --wikimapper-dir data/wikimapper
```

Relancer uniquement l'extraction du CSV (DB déjà construite) :

```bash
python utils/extract_all_entities.py --db data/wikimapper/index_enwiki-latest.db
```

Résultat : `data/wikidata_qid_titles.csv` — deux colonnes (`qid`, `title`), ~6-7M lignes, dédupliqué par QID.

---

### Étape 2 : Construire les index de candidats

**BM25 (sparse retrieval)** — quelques secondes sur CPU :

```bash
python candidates/bm25_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/bm25.pkl
```

**Dense retrieval** — nécessite GPU, environ 34 heures :

```bash
python candidates/dense_retriever.py --titles data/wikidata_qid_titles.csv --out candidates/index/
```

Les index sont sauvegardés sur disque et chargés instantanément par la suite.

---

### Étape 3 : Pré-calculer les candidats

Pré-calculer les candidats pour toutes les mentions ELGOLD permet d'éviter de recharger le retriever à chaque évaluation.
Le script d'évaluation détecte automatiquement les fichiers pré-calculés présents dans `candidates/precomputed/`.

```bash
# BM25 candidates
python candidates/precompute_candidates.py --retriever bm25 --k 100

# Dense candidates
python candidates/precompute_candidates.py --retriever dense --k 64
```

Résultats attendus :

```text
candidates/precomputed/bm25_k100.json
candidates/precomputed/dense_k64.json
```

Remarque :

* `bm25_k100.json` contient les top-100 candidats BM25 par mention.
* `dense_k64.json` contient les top-64 candidats Dense par mention.
* Si ces fichiers existent, `evaluation/eval_accuracy.py` les charge automatiquement.
* S'ils n'existent pas, le script lance le retrieval en direct à partir des index dans `candidates/index/`.

---

### Étape 4 : Lancer l'évaluation

Le pipeline d'évaluation est :

```text
Retriever candidates → optional Reranker → MCQ Prompt → LLM → Parsing → Metrics
```

Par défaut, le reranker est activé.

#### BM25 + Reranker + LLM

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --batch-size 8
```

Avec cette configuration :

```text
BM25 top-100 → Qwen3-Reranker top-10 → Qwen3-4B LLM
```

Le reranker utilise par défaut :

```text
Qwen/Qwen3-Reranker-4B
```

Le nombre de candidats conservés après reranking est contrôlé par :

```bash
--k 10
```

Exemple :

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

Avec cette configuration :

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

Avec cette configuration :

```text
BM25 top-100 + Dense top-64 → union candidates → Qwen3-Reranker top-10 → Qwen3-4B LLM
```

La fusion fusionne les candidats BM25 et Dense en supprimant les doublons, en gardant l'ordre BM25 puis les ajouts Dense.

---

### Évaluation sans reranker

Pour désactiver le reranker :

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --no-reranker \
    --batch-size 8
```

Important :

* Avec reranker : `--k` limite le nombre de candidats après reranking.
* Sans reranker : tous les candidats récupérés sont passés au LLM, sans tronquage au top-k.
* Donc, en mode `--no-reranker`, `--k` est ignoré pour la sélection des candidats.

Exemple :

```text
BM25 avec --no-reranker:
BM25 top-100 → Qwen3-4B LLM

Dense avec --no-reranker:
Dense top-64 → Qwen3-4B LLM

Fusion avec --no-reranker:
BM25 top-100 + Dense top-64 → union candidates → Qwen3-4B LLM
```

---

### Utilisation des candidats pré-calculés

Aucun flag spécifique n'est nécessaire.

Le script cherche automatiquement :

```text
candidates/precomputed/bm25_k*.json
candidates/precomputed/dense_k*.json
```

Selon le mode choisi :

| Mode              | Fichier pré-calculé utilisé            |
| ----------------- | -------------------------------------- |
| `zero_shot_bm25`  | `candidates/precomputed/bm25_k*.json`  |
| `zero_shot_dense` | `candidates/precomputed/dense_k*.json` |
| `fusion`          | Les deux fichiers BM25 et Dense        |

Si plusieurs fichiers correspondent au même pattern, le dernier fichier trié est utilisé.

---

### Options utiles

| Flag                                | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `--mode zero_shot_bm25`             | Évaluation avec candidats BM25                           |
| `--mode zero_shot_dense`            | Évaluation avec candidats Dense                          |
| `--mode fusion`                     | Fusion des candidats BM25 et Dense                       |
| `--model Qwen/Qwen3-4B`             | Modèle LLM utilisé pour choisir la réponse finale        |
| `--reranker Qwen/Qwen3-Reranker-4B` | Modèle de reranking, utilisé sauf si `--no-reranker`     |
| `--no-reranker`                     | Désactive le reranker et passe tous les candidats au LLM |
| `--k 10`                            | Nombre de candidats conservés après reranking            |
| `--batch-size 8`                    | Taille du batch LLM                                      |
| `--max-new-tokens 2048`             | Nombre maximal de tokens générés                         |
| `--max-examples 200`                | Limiter le benchmark pour tester rapidement              |
| `--resume`                          | Reprendre depuis le checkpoint                           |

---

### Résultats

Les résultats sont sauvegardés dans :

```text
evaluation/results/{mode}_{model}_{timestamp}.csv
evaluation/results/{mode}_{model}_{timestamp}_metrics.json
```

Le fichier CSV contient notamment :

| Colonne                | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `mention`              | Mention ELGOLD                                                      |
| `gold`                 | Entité correcte                                                     |
| `candidates_retrieval` | Candidats récupérés par BM25/Dense/Fusion                           |
| `candidates_reranked`  | Candidats après reranking, ou tous les candidats si `--no-reranker` |
| `gold_in_retrieval`    | Indique si le gold est dans les candidats récupérés                 |
| `gold_in_reranked`     | Indique si le gold est dans les candidats passés au LLM             |
| `predicted`            | Entité prédite par le LLM                                           |
| `correct`              | Indique si la prédiction est correcte                               |
| `raw_output`           | Sortie brute du LLM                                                 |
| `error`                | Erreur éventuelle ou sortie non parsable                            |

---

### Reprise après interruption

Si un job est interrompu, un checkpoint est sauvegardé automatiquement dans `evaluation/results/`.

Relancer avec :

```bash
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --resume
```

Le script reprend les exemples déjà traités depuis le checkpoint.


### Étape 5 : Analyser les résultats

```bash
jupyter notebook notebooks/zero_shot_benchmark.ipynb
```

Le notebook permet de visualiser les résultats, inspecter les erreurs, et comparer les configurations (BM25 vs Dense vs Fusion).

---

## Structure du projet

```
├── candidates/
│   ├── __init__.py
│   ├── bm25_retriever.py           # Retrieval sparse BM25
│   ├── dense_retriever.py          # Retrieval dense Qwen3-Embedding + FAISS
│   ├── reranker.py                 # Reranking des candidats avec Qwen3-Reranker
│   ├── precompute_candidates.py    # Pré-calcul des candidats BM25/Dense
│   ├── index/                      # Index sérialisés (.pkl, .faiss) — gitignored
│   └── precomputed/                # Candidats pré-calculés JSON
│
├── data/
│   ├── benchmarks/                 # Fichiers ELGOLD (.txt annotés)
│   ├── wikimapper/                 # Dump Wikipedia + base SQLite — gitignored
│   └── wikidata_qid_titles.csv     # Tous les titres Wikipedia + QIDs — gitignored
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_accuracy.py            # Script principal d'évaluation (cluster)
│   └── results/                    # CSV + JSON des résultats — gitignored
│
├── notebooks/
│   └── zero_shot_benchmark.ipynb
│
├── slurm/                          # Scripts SLURM pour le cluster
│   ├── run_eval_bm25.sh
│   ├── run_eval_dense.sh
│   └── run_precompute_bm25.sh
│
└── utils/
    ├── __init__.py
    ├── elgold_parser.py            # Parsing du benchmark ELGOLD
    ├── extract_all_entities.py     # Extraction des entités via wikimapper
    ├── llm.py                      # Chargement LLM + génération batch
    ├── metrics.py                  # Accuracy, Recall@K, normalisation
    ├── prompt_builder.py           # Construction du prompt MCQ (indices numériques)
    └── response_parser.py          # Parsing de la réponse LLM (index → titre)
```

---

## Référence des scripts

### `utils/extract_all_entities.py`

Extrait tous les titres Wikipedia + QIDs Wikidata depuis la base wikimapper. Déduplique par QID (garde le titre canonique).

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

### `candidates/precompute_candidates.py`

Pré-calcule les candidats pour toutes les mentions du benchmark ELGOLD et sauvegarde en JSON.

```
--retriever TYPE        bm25 | dense
--k N                   Nombre de candidats par mention (défaut: 100)
--bm25-index PATH       Chemin vers l'index BM25
--dense-index PATH      Chemin vers l'index dense
--out PATH              Fichier JSON de sortie (auto-nommé si omis)
```

### `evaluation/eval_accuracy.py`

Script principal d'évaluation zero-shot sur ELGOLD.

Pipeline complet :

```text
candidate retrieval → optional reranking → MCQ prompt → LLM generation → parsing → metrics
```

Modes disponibles :

```text
zero_shot_bm25
zero_shot_dense
fusion
```

Arguments principaux :

```text
--mode MODE             zero_shot_bm25 | zero_shot_dense | fusion
--model NAME            Modèle HuggingFace utilisé comme LLM final
--reranker NAME         Modèle HuggingFace utilisé comme reranker
--no-reranker           Désactive le reranker
--k N                   Nombre de candidats gardés après reranking
--batch-size N          Taille du batch LLM
--max-new-tokens N      Nombre maximal de tokens générés
--max-examples N        Limiter le benchmark pour tester
--index PATH            Chemin vers l'index BM25
--dense-index PATH      Chemin vers l'index Dense
--benchmarks PATH       Chemin vers les fichiers ELGOLD
--results-dir PATH      Dossier de sauvegarde des résultats
--resume                Reprendre depuis le checkpoint
```

Fonctionnement des candidats pré-calculés :

* Le script cherche automatiquement les fichiers dans `candidates/precomputed/`.
* En mode BM25, il cherche `bm25_k*.json`.
* En mode Dense, il cherche `dense_k*.json`.
* En mode Fusion, il cherche les deux.
* Si aucun fichier pré-calculé n'est trouvé, le retriever est chargé et les candidats sont calculés en direct.

Fonctionnement du reranker :

* Le reranker est activé par défaut.
* Il reçoit les candidats récupérés par BM25, Dense ou Fusion.
* Il réordonne les candidats et garde le top-`k`.
* Le LLM reçoit seulement les candidats rerankés.
* Avec `--no-reranker`, le LLM reçoit tous les candidats récupérés, sans tronquage.

Exemples :

```bash
# BM25 + reranker
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --k 10

# Dense + reranker
python evaluation/eval_accuracy.py \
    --mode zero_shot_dense \
    --model Qwen/Qwen3-4B \
    --k 10

# Fusion + reranker
python evaluation/eval_accuracy.py \
    --mode fusion \
    --model Qwen/Qwen3-4B \
    --k 10

# BM25 sans reranker
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --no-reranker
```

### `utils/llm.py`

Fonctions LLM partagées entre le notebook et le script d'évaluation.

- `load_model(name)` — charge le modèle + tokenizer
- `generate_batch(prompts, tokenizer, model, max_new_tokens)` — génère une réponse par prompt, en batch

### `utils/elgold_parser.py`

Parsing du benchmark ELGOLD.

- `parse_elgold_file(path)` — extrait les mentions annotées d'un fichier `.txt`
- `load_benchmark(dir, max_examples)` — charge tous les fichiers et retourne un DataFrame

### `utils/prompt_builder.py`

Construit le prompt MCQ : contexte (mention entre crochets) + type NER + candidats numérotés `0. Titre`. Le LLM répond `"answer: 3"`.

### `utils/response_parser.py`

Parse la réponse du LLM : cherche le pattern `answer: N`, puis un index en début de ligne, puis un index isolé. Gère les blocs `<think>` et les guillemets.

### `utils/metrics.py`

Métriques : `normalize_entity()`, `is_correct()`, `recall_at_k()`, `compute_metrics()`, `print_metrics()`.
>>>>>>> b7ca8ba3ef58080e4930f52e49f32e6ae3235a24
