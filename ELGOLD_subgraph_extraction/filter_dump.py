import sys, gzip, json

# Charge tes QIDs cibles
import json as j
with open("elgold_entities_per_domain.json") as f:
    elgold = j.load(f)
target_qids = set()
for cat in elgold.values():
    target_qids.update(cat["entities_qids"])

print(f"Recherche de {len(target_qids)} QIDs...", file=sys.stderr)

results = {}
buf = sys.stdin.buffer  # lit le flux curl

with gzip.open(buf, "rt", encoding="utf-8") as f:
    f.read(2)  # skip "[\n"
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
            print(f"  ✓ {qid} ({len(results)}/{len(target_qids)})", file=sys.stderr)
            if len(results) == len(target_qids):
                break  # stop dès qu'on a tout

with open("wikidata_elgold.json", "w") as out:
    json.dump(results, out)

print("✓ Sauvegardé dans wikidata_elgold.json", file=sys.stderr)