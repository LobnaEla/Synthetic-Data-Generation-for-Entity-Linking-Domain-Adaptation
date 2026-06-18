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

Pré-calculer les candidats pour toutes les mentions ELGOLD (évite de recharger le retriever à chaque évaluation) :

```bash
python candidates/precompute_candidates.py --retriever bm25 --k 100
python candidates/precompute_candidates.py --retriever dense --k 64
```

Résultat : `candidates/precomputed/bm25_k100.json` et `dense_k64.json`.

---

### Étape 4 : Lancer l'évaluation

**Avec candidats pré-calculés (recommandé) :**

```bash
# BM25
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \

# Dense
python evaluation/eval_accuracy.py \
    --mode zero_shot_dense \
    --model Qwen/Qwen3-4B \
```

**Sans candidats pré-calculés (retrieval live) :**

```bash
python evaluation/eval_accuracy.py --mode zero_shot_bm25 --model Qwen/Qwen3-4B
python evaluation/eval_accuracy.py --mode zero_shot_dense --model Qwen/Qwen3-4B
```

Options :

| Flag | Description |
|---|---|
| `--batch-size 8` | Taille du batch LLM (ajuster selon la VRAM) |
| `--max-new-tokens 2048` | Tokens max (défaut: 2048) |
| `--max-examples 200` | Limiter le nombre d'exemples (pour tester) |
| `--candidates PATH` | Candidats pré-calculés (JSON) |
| `--resume` | Reprendre depuis le dernier checkpoint |

Les résultats sont sauvegardés dans :
- `evaluation/results/{mode}_{model}_{timestamp}.csv` — prédictions détaillées
- `evaluation/results/{mode}_{model}_{timestamp}_metrics.json` — métriques agrégées

**Reprise après interruption :**

```bash
# Le job a été interrompu → un checkpoint existe
# Relancer avec --resume pour continuer où ça s'est arrêté
python evaluation/eval_accuracy.py \
    --mode zero_shot_bm25 \
    --model Qwen/Qwen3-4B \
    --candidates candidates/precomputed/bm25_k100.json \
    --resume
```

---

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
│   ├── bm25_retriever.py           # Retrieval sparse (BM25) — build + query
│   ├── dense_retriever.py          # Retrieval dense (Qwen3-Embedding + FAISS) — build + query
│   ├── precompute_candidates.py    # Pré-calcul des candidats pour le benchmark
│   ├── index/                      # Index sérialisés (bm25.pkl, dense.faiss) — gitignored
│   └── precomputed/                # Candidats pré-calculés (JSON) — gitignored
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

Évaluation complète : candidats → prompt MCQ → LLM → parsing → métriques. Supporte le checkpoint/resume.

```
--mode MODE             zero_shot_bm25 | zero_shot_dense
--model NAME            Modèle HuggingFace (ex: Qwen/Qwen3-4B)
--candidates PATH       Candidats pré-calculés JSON (optionnel)
--batch-size N          Taille du batch LLM (défaut: 8)
--max-new-tokens N      Tokens max (défaut: 2048)
--max-examples N        Limiter le benchmark (pour tester)
--resume                Reprendre depuis le checkpoint
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