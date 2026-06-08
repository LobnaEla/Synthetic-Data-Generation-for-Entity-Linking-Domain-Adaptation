import urllib.request, gzip, json

url = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz"

with open("elgold_entities_per_domain.json", encoding="utf-8") as f:
    elgold = json.load(f)
target_qids = set()
for cat in elgold.values():
    target_qids.update(cat["entities_qids"])
print(f"Recherche de {len(target_qids)} QIDs...")

results = {}
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=3600) as response:
    with gzip.open(response, "rt", encoding="utf-8") as f:
        f.read(2)
        for line in f:
            line = line.strip().rstrip(",")
            if not line or line == "]":
                continue
            try:
                entity = json.loads(line)
            except:
                continue
            qid = entity.get("id", "")
            if qid in target_qids:
                results[qid] = entity
                print(f"  ✓ {qid} ({len(results)}/{len(target_qids)})")
                if len(results) == len(target_qids):
                    break

with open("wikidata_elgold.json", "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False)
print("✓ Sauvegardé dans wikidata_elgold.json")




