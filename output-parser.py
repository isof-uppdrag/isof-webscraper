import json
import os
from collections import defaultdict

# This program sorts the scraped texts based on language into separate per-langauge json files

INPUT_FILE = os.path.join("output", "scraped-corpus.json")
OUTPUT_DIR = os.path.join("output", "corpora sorted by language")


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select the metadata that should be present in the new file
FIELDS = [
    "url",
    "category",
    "lang-fasttext-identified",
    "lang-fasttext-confidence",
    "lang_prediction_level",
    "sentence_lang_distribution",
    "final_prediction",
    "classification_type",
    "crawl_timestamp",
    "published",
    "title",
    "text"
]

by_language = defaultdict(list)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)
        lang = record.get("final_prediction", "unknown")

        filtered = {field: record.get(field) for field in FIELDS}
        by_language[lang].append(filtered)

for lang, records in by_language.items():
    out_path = os.path.join(OUTPUT_DIR, f"{lang}.json")
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(records, out, ensure_ascii=False, indent=2)

print(f"Created {len(by_language)} language files in '{OUTPUT_DIR}'")