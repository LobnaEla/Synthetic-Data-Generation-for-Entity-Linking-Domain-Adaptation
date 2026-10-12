# Synthetic Data for Entity Linking Domain Adaptation

Ce projet étudie la génération automatique de données annotées pour
l'entity linking. L'objectif est de produire des exemples synthétiques à
partir de sous-graphes Wikidata, puis de comparer leur utilisation à un
entraînement sur des annotations humaines et à une approche zero-shot.

## Vue d'ensemble

Le protocole est évalué sur le benchmark [ELGOLD](https://doi.org/10.34808/f3q9-9k64),
réparti en sept domaines :

- `News`
- `Job_offers`
- `Movie_reviews`
- `Automotive_blogs`
- `Amazon_product_reviews`
- `Scientific_abstracts`
- `Historic_blogs`

Le pipeline compare trois branches :

1. **Zero-shot / few-shot** : prompting d'un modèle de langage sur ELGOLD.
2. **Fine-tuning humain** : entraînement sur des mentions ZELDA filtrées par
   les entités présentes dans ELGOLD.
3. **Fine-tuning synthétique** : génération de textes et d'annotations depuis
   des sous-graphes Wikidata, puis entraînement et évaluation sur ELGOLD.

Les entités sont identifiées par leur **Wikidata Q-ID**, ce qui fournit un
identifiant commun entre les différentes branches.

## Organisation du dépôt

```text
benchmark_pipeline/
  candidates/              Retrieval BM25, dense et reranking
  data/                    Données et statistiques ELGOLD
  evaluation/              Évaluation zero-shot sur ELGOLD
  notebooks/               Analyse des résultats
  scripts/                 Outils d'analyse
  utils/                   Parsing, prompts, modèles et métriques

fine_tuning/
  filter_zelda*.py         Construction des jeux d'entraînement
  make_splits.py           Splits train/validation versionnés
  finetune*.py             Entraînement QLoRA
  evaluate_elgold.py       Évaluation des adaptateurs
  coverage_report.py       Couverture des entités ELGOLD

subgraph_extraction/
  fetch_entities.py        Extraction des entités et relations
  edges_*.csv              Arêtes par domaine
  nodes_*.csv              Noeuds par domaine
  extraction.ipynb         Exploration de l'extraction

subgraph_generation/
  Génération des sous-graphes et artefacts associés

text_generation/
  Notebooks de génération, aplatissement et regroupement des corpus
  final dataset/            Jeux de données synthétiques JSONL
```

## Installation

Les dépendances Python regroupées dans le fichier `requirements.txt` à la racine du projet :

```bash
pip install -r requirements.txt
```

Pour le fine-tuning avec Unsloth et CUDA, réinstallez ensuite la version de
PyTorch adaptée à votre installation :

```powershell
pip install --force-reinstall torch==2.10.0+cu128 torchvision==0.25.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

Les étapes de fine-tuning nécessitent une GPU compatible CUDA et suffisamment
de mémoire. Les notebooks nécessitent également Jupyter.

## 1. Génération des données synthétiques

Les notebooks du dossier `text_generation/` couvrent notamment :

- la préparation du corpus et son aplatissement ;
- la génération de textes avec Qwen ou Groq ;
- la construction d'exemples d'entity linking ;
- le regroupement des jeux de données par domaine.

Les jeux finaux sont disponibles dans `text_generation/final dataset/` au
format JSONL, un fichier par domaine.

## 2. Extraction des sous-graphes

Le dossier `subgraph_extraction/` contient les outils d'extraction des noeuds
et des relations depuis Wikidata, ainsi que les exports CSV par domaine.

L'exploration principale se trouve dans :

```text
subgraph_extraction/extraction.ipynb
```

Les fichiers `nodes_*.csv` et `edges_*.csv` servent ensuite de base à la
génération des contextes synthétiques.

## 3. Fine-tuning sur les données humaines ou synthétiques

### Préparer les entités ELGOLD

```powershell
cd fine_tuning
python extract_elgold_entities.py <dossier_elgold> elgold_entities_per_domain.json `
  --wikimapper_db <chemin_vers_index_enwiki-latest.db>
```

### Construire les jeux d'entraînement ZELDA

Pour l'approche générative :

```powershell
python filter_zelda.py <zelda_train.jsonl> elgold_entities_per_domain.json `
  --wikimapper_db <chemin_vers_index_enwiki-latest.db> `
  --output zelda_filtered_train.jsonl
```

Pour l'approche reranking :

```powershell
python filter_zelda_rerank.py <zelda_train.jsonl> elgold_entities_per_domain.json `
  <alias_table.pickle> <entity_descriptions.jsonl> `
  --wikimapper_db <chemin_vers_index_enwiki-latest.db> `
  --output zelda_rerank_train.jsonl
```

Créer ensuite les splits déterministes :

```powershell
python make_splits.py zelda_filtered_train.jsonl splits/generative
python make_splits.py zelda_rerank_train.jsonl splits/rerank
```

Le split utilise une validation de 1 % et produit un manifeste avec les
empreintes SHA-256 des fichiers.

### Entraîner un adaptateur QLoRA

```powershell
python -u finetune_unsloth.py splits/generative/train.jsonl output_generative `
  --val_data splits/generative/val.jsonl --epochs 1 --report_to none
```

Pour reprendre un checkpoint, utiliser `--resume`. Pour un test rapide,
utiliser `--max_steps`.

### Évaluer sur ELGOLD

Mode génératif :

```powershell
python -u evaluate_elgold.py <dossier_elgold> <adaptateur_lora> `
  elgold_entities_per_domain.json --mode generative `
  --train_jsonl splits/generative/train.jsonl `
  --output_prefix eval_generative
```

Mode reranking : ajouter `--alias_pickle`, `--descriptions_jsonl` et
`--wikimapper_db` afin de reconstruire la même liste de candidats qu'à
l'entraînement.

Le rapport fournit notamment l'accuracy@1 globale et par domaine, ainsi que
les performances sur les entités vues et non vues à l'entraînement.

## 4. Benchmark zero-shot

### Construire la base d'entités Wikipedia

```powershell
cd benchmark_pipeline
python utils/extract_all_entities.py --setup --wikimapper-dir data/wikimapper
```

Une fois la base créée, l'extraction seule peut être relancée avec :

```powershell
python utils/extract_all_entities.py --db data/wikimapper/index_enwiki-latest.db
```

Le résultat est `data/wikidata_qid_titles.csv`.

### Construire les index et les candidats

```powershell
python candidates/bm25_retriever.py `
  --titles data/wikidata_qid_titles.csv --out candidates/index/bm25.pkl

python candidates/dense_retriever.py `
  --titles data/wikidata_qid_titles.csv --out candidates/index/

python candidates/precompute_candidates.py --retriever bm25 --k 100
python candidates/precompute_candidates.py --retriever dense --k 64
```

L'index dense est beaucoup plus coûteux à construire et nécessite une GPU.
Les fichiers pré-calculés sont automatiquement détectés dans
`candidates/precomputed/`.

### Lancer une évaluation

BM25, dense ou fusion des deux :

```powershell
python evaluation/eval_accuracy.py `
  --mode zero_shot_bm25 --model Qwen/Qwen3-4B --k 10 --batch-size 8

python evaluation/eval_accuracy.py `
  --mode zero_shot_dense --model Qwen/Qwen3-4B --k 10 --batch-size 8

python evaluation/eval_accuracy.py `
  --mode fusion --model Qwen/Qwen3-4B --k 10 --batch-size 8
```

Le pipeline est :

```text
retrieval -> reranking optionnel -> prompt MCQ -> LLM -> parsing -> métriques
```

Le reranker par défaut est `Qwen/Qwen3-Reranker-4B`. Pour le désactiver :

```powershell
python evaluation/eval_accuracy.py `
  --mode zero_shot_bm25 --model Qwen/Qwen3-4B --no-reranker
```

Pour séparer le reranking et l'appel au LLM :

```powershell
python evaluation/eval_accuracy.py --mode zero_shot_bm25 --stage rerank --k 10
python evaluation/eval_accuracy.py --mode zero_shot_bm25 --stage llm `
  --model Qwen/Qwen3-4B --k 10 --batch-size 8
```

Les résultats sont enregistrés dans `benchmark_pipeline/evaluation/results/`.
L'option `--domains News,Job_offers` permet de limiter l'évaluation à certains
domaines, et `--resume` permet de reprendre une exécution interrompue.

L'analyse des résultats est disponible dans :

```powershell
jupyter notebook notebooks/zero_shot_benchmark.ipynb
```

## Résultats de référence

Les expériences de fine-tuning reranking rapportées dans le projet donnent une
accuracy@1 pondérée de **69,2 %** sur ELGOLD, avec une couverture des entités
ELGOLD de **79,9 %** dans le jeu d'entraînement humain filtré.

Ces chiffres dépendent du modèle, des données disponibles et des paramètres
d'exécution. Les fichiers de résultats et les modèles générés ne sont pas tous
versionnés dans Git.

## Données et fichiers générés

Les artefacts volumineux sont généralement exclus du versionnement :

- base Wikimapper et dump Wikipedia ;
- index BM25, FAISS et embeddings ;
- checkpoints et adaptateurs LoRA ;
- résultats d'évaluation ;
- jeux ZELDA filtrés et splits générés.

Vérifier les chemins d'entrée avant chaque commande et conserver les ressources
externes dans des dossiers séparés des scripts versionnés.

## Références

- [ELGOLD](https://doi.org/10.34808/f3q9-9k64)
- [ZELDA](https://github.com/flairNLP/zelda)
- [Qwen](https://huggingface.co/Qwen)
- [Unsloth](https://github.com/unslothai/unsloth)
- [Wikimapper](https://github.com/jcklie/wikimapper)