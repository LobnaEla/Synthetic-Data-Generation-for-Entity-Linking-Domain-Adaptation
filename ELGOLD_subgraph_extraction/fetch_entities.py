import requests, json, time

with open("elgold_entities_per_domain.json", encoding="utf-8") as f:
    elgold = json.load(f)
target_qids = list(set(
    qid for cat in elgold.values() for qid in cat["entities_qids"]
))
print(f"{len(target_qids)} QIDs à récupérer")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ELgoldProject/1.0 (student; wikidata-api)"})

results = {}
BATCH = 50

for i in range(0, len(target_qids), BATCH):
    batch = target_qids[i:i+BATCH]
    success = False
    for attempt in range(3):  # 3 tentatives par batch
        try:
            r = SESSION.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "format": "json",
                    "languages": "en",
                    "props": "labels|claims",
                },
                timeout=60
            )
            print(f"  HTTP {r.status_code} | taille réponse : {len(r.content)} bytes")
            data = r.json()
            for qid, entity in data.get("entities", {}).items():
                results[qid] = entity
            print(f"  ✓ {min(i+BATCH, len(target_qids))}/{len(target_qids)}")
            success = True
            break
        except Exception as e:
            print(f"  ⚠ tentative {attempt+1} échouée : {e}")
            time.sleep(10)
    
    if not success:
        print(f"  ✗ batch {i//BATCH} abandonné")
    
    time.sleep(1)

    if i % 200 < BATCH:
        with open("wikidata_elgold.json", "w", encoding="utf-8") as out:
            json.dump(results, out, ensure_ascii=False)
        print(f"  💾 Sauvegarde ({len(results)} entités)")

with open("wikidata_elgold.json", "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False)
print("✓ wikidata_elgold.json prêt !")