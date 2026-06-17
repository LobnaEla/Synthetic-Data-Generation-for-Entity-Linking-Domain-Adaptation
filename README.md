# Synthetic Data for Entity Linking Domain Adaptation

Projet DSAI 2026 — groupe 13.

## Contexte

L'objectif global du projet est de construire un système capable de **générer
automatiquement des données annotées** pour l'entity linking (EL), suffisamment
fidèles pour entraîner un modèle dont les performances approchent celles d'un
modèle entraîné sur des annotations humaines.

La validation repose sur un **benchmark à trois branches**, toutes évaluées sur
le même benchmark ELGOLD :

| Branche | Données d'entraînement | Rôle |
|---|---|---|
| (1) Few-shot | aucune (prompting) | plancher de performance |
| (2) Fine-tuning humain | ZELDA filtré (**ce dépôt**) | référence haute |
| (3) Fine-tuning synthétique | génération depuis subgraph Wikidata | doit converger vers (2) |

Ce dépôt implémente la **branche (2)** : fine-tuning de
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
(via [Unsloth](https://github.com/unslothai/unsloth) + QLoRA) sur des mentions
issues du corpus annoté humain [ZELDA](https://github.com/flairNLP/zelda),
filtrées pour correspondre aux entités du benchmark
[ELGOLD](https://doi.org/10.34808/f3q9-9k64).

## Décisions d'architecture

| Décision | Choix |
|---|---|
| Identifiant d'entité pivot | **Wikidata Q-ID** (ex : `Q312`) — stable, commun aux branches (2) et (3) |
| Mapping Wikipedia  Wikidata | [`wikimapper`](https://github.com/jcklie/wikimapper) (DB SQLite offline) |
| Granularité d'entraînement | **mention-level** : 1 exemple = 1 mention dans son contexte |
| Stratégie | **1 modèle multi-domaines** (pas un modèle par domaine) |
| Deux approches EL comparées | **générative** (produire le Q-ID) et **reranking** (choisir parmi K candidats) |
| Test set | **ELGOLD uniquement** — aucun test ZELDA réservé ; le val (1 %) sert au monitoring |
| Mentions NIL | exclues du scoring à l'éval (comptées à part) |

## Pipeline

```
ELGOLD (data/)                ZELDA train (94 979 docs, 2,6 M mentions)
      │                                  │
      ▼                                  ▼
extract_elgold_entities.py    filter_zelda.py  /  filter_zelda_rerank.py
      │                                  │
      ▼                                  ▼
elgold_entities_per_domain.json   zelda_filtered_train.jsonl (94 097 ex.)
      │                           zelda_rerank_train.jsonl   (94 097 ex.)
      │                                  │
      │                                  ▼
      │                           make_splits.py → splits/{generative,rerank}/
      │                                  │         (train 93 157 / val 940 + manifest SHA-256)
      ├──────────► coverage_report.py ◄──┤
      │                                  ▼
      │                           finetune_unsloth.py → adaptateur LoRA
      │                                  │
      └──────────► evaluate_elgold.py ◄──┘
                          │
                          ▼
            accuracy@1 par domaine + global
```

### 1. `extract_elgold_entities.py` — entités cibles par domaine

Parse les annotations inline ELGOLD `{{mention | classe NER | titre Wikipedia}}`,
groupe par domaine (préfixe du nom de fichier : `1_*` News, `2_*` Job offers,
`3_*` Movie reviews, `4_*` Automotive blogs, `5_*` Amazon reviews, `6a-6e_*`
Scientific abstracts, `8_*` Historic blogs), puis mappe chaque titre vers son
Q-ID via wikimapper.

```powershell
python ProjectDSAI2026_13\extract_elgold_entities.py data elgold_entities_per_domain.json `
    --wikimapper_db wikimapper_data\index_enwiki-latest.db
```

Résultat obtenu : 276 docs, 3 106 mentions liées + 453 NIL, **1 567 titres
uniques mappés à 100 %** vers 1 447 Q-IDs distincts.

### 2. `filter_zelda.py` — dataset génératif

Pour chaque mention ZELDA dont le Q-ID appartient aux entités ELGOLD, produit
un exemple `{prompt, ground_truth_id, domain, mention}`. Le modèle doit
**générer le Q-ID** directement.

**Prompt utilisé (génératif)** — contexte ±512 caractères :

```
Identify the Wikidata Q-ID of the entity referred to by the mention marked
with **double asterisks** in the following text.

Text:
{contexte avant}**{mention}**{contexte après}

Answer with only the Q-ID (e.g. Q42).
```

Completion attendue : le Q-ID brut (ex. `Q1108617`).

```powershell
python ProjectDSAI2026_13\filter_zelda.py zelda\train_data\zelda_train.jsonl `
    elgold_entities_per_domain.json `
    --wikimapper_db wikimapper_data\index_enwiki-latest.db `
    --output zelda_filtered_train.jsonl
```

### 3. `filter_zelda_rerank.py` — dataset reranking

Même filtrage, mais le prompt liste **K=10 candidats** issus de l'alias table
de ZELDA (`zelda_mention_entities_counter.pickle`, triés par fréquence) avec
une description (≤150 caractères, `entity_descriptions.jsonl`), plus une option
`NONE`. Le modèle doit **choisir** parmi les candidats.

**Prompt utilisé (reranking)** — contexte ±384 caractères :

```
Identify the Wikidata entity for the mention marked with **double asterisks**
in the context.

Context:
{contexte avant}**{mention}**{contexte après}

Candidates:
  Q41112 — Schizophrenia: Schizophrenia is a mental illness characterized by...
  Q1967324 — Sluggish schizophrenia: Sluggish schizophrenia or slow progressi...
  ...
  NONE — none of the candidates above

Which candidate does the mention refer to? Answer with only the Q-ID (e.g. Q42) or "NONE".
```

Completion attendue : le Q-ID du bon candidat, ou `NONE` si le gold n'est pas
dans la liste (1,3 % des cas — recall@10 de l'alias table : **98,7 %**).

```powershell
python ProjectDSAI2026_13\filter_zelda_rerank.py zelda\train_data\zelda_train.jsonl `
    elgold_entities_per_domain.json `
    zelda\other\zelda_mention_entities_counter.pickle `
    zelda\other\entity_descriptions.jsonl `
    --wikimapper_db wikimapper_data\index_enwiki-latest.db `
    --output zelda_rerank_train.jsonl
```

### 4. `make_splits.py` — splits versionnés

Split train/val déterministe (seed 42, val = 1 %). Écrit un `manifest.json`
avec comptes par domaine et **hash SHA-256** des fichiers : la branche (3)
peut vérifier qu'elle rejoue exactement le même protocole.
Pas de test ZELDA : le test set du protocole est ELGOLD.

```powershell
python ProjectDSAI2026_13\make_splits.py zelda_filtered_train.jsonl splits\generative
python ProjectDSAI2026_13\make_splits.py zelda_rerank_train.jsonl splits\rerank
```

### 5. `coverage_report.py` — couverture ELGOLD ↔ train

Mesure, par domaine, la fraction des entités gold d'ELGOLD vues au moins une
fois dans le train filtré. Résultat clé : **79,9 % de couverture globale**
(291 entités jamais vues → zero-shot à l'éval). Détail par domaine : de 68,3 %
(Scientific abstracts) à 90,9 % (Movie reviews).

```powershell
python ProjectDSAI2026_13\coverage_report.py splits\generative\train.jsonl `
    ProjectDSAI2026_13\elgold_entities_per_domain.json --output coverage_report.json
```

### 6. `finetune_unsloth.py` — entraînement QLoRA

- Modèle : `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit` (pré-quantizé 4-bit, variante non-thinking).
- LoRA : r=128, alpha=256, dropout 0, cibles `q/k/v/o/gate/up/down_proj`,
  `use_gradient_checkpointing="unsloth"`.
- Chat template Qwen3 + `train_on_responses_only` (la loss ne porte que sur la
  réponse de l'assistant, pas sur le prompt).
- LR 1e-4 cosine, batch 2 × accumulation 8 (batch effectif 16), bf16.
- `--val_data` pour utiliser le val versionné ; `--resume` pour reprendre d'un
  checkpoint ; `--max_steps` pour les smoke tests ; `--report_to none` sans W&B.

```powershell
$env:CC = "$env:APPDATA\Python\Python313\site-packages\triton\runtime\tcc\tcc.exe"
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python -u ProjectDSAI2026_13\finetune_unsloth.py splits\generative\train.jsonl output_generative `
    --val_data splits\generative\val.jsonl --epochs 1 --report_to none
```

### 7. `evaluate_elgold.py` — évaluation sur ELGOLD

Construit les prompts de test avec **les mêmes templates que l'entraînement**
(imports directs depuis les scripts de filtrage — aucune copie qui pourrait
diverger), inférence greedy par batch, puis accuracy@1 par domaine et globale.

Protocole figé (à rejouer à l'identique pour la branche (3)) :
- NIL exclus du scoring ; correct ⟺ Q-ID prédit == Q-ID gold ;
- `NONE` ou sortie imparseable = incorrect ;
- décodage greedy, `max_new_tokens=16` ;
- option `--train_jsonl` : décompose l'accuracy en entités **vues / jamais
  vues** à l'entraînement (à croiser avec le rapport de couverture).

```powershell
python -u ProjectDSAI2026_13\evaluate_elgold.py data output_generative\final_adapter `
    ProjectDSAI2026_13\elgold_entities_per_domain.json --mode generative `
    --train_jsonl splits\generative\train.jsonl --output_prefix eval_generative
```

## Installation

```powershell
pip install unsloth wandb wikimapper
# Unsloth remplace torch par une build CPU : réinstaller la build CUDA ensuite
pip install --force-reinstall torch==2.10.0+cu128 torchvision==0.25.0+cu128 `
    --index-url https://download.pytorch.org/whl/cu128

# Base wikimapper (une fois, ~30 min)
wikimapper download enwiki-latest --dir wikimapper_data
wikimapper create enwiki-latest --dumpdir wikimapper_data `
    --target wikimapper_data\index_enwiki-latest.db
```


## État d'avancement

- [x] Extraction entités ELGOLD (100 % mappées)
- [x] Filtrage ZELDA génératif + reranking (94 097 exemples chacun)
- [x] Splits versionnés + manifests SHA-256
- [x] Rapport de couverture (79,9 % global)
- [x] Smoke test entraînement validé sur RTX 5090 Laptop (loss 8,9 → 0,5 sur 665 steps)
- [ ] Run complet 1 epoch génératif (interrompu au step 665/5823 — crash machine, à relancer avec `--save_steps` réduit)
- [ ] Run complet reranking
- [ ] Évaluation ELGOLD des deux adaptateurs
- [ ] `requirements.txt` figé + rapport final
