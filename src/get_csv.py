# src/get_csv.py
# Create per-domain CSVs your app can ingest directly.
# Run with: python src/get_csv.py
# Dependencies: pip install datasets pandas

import os
from datasets import load_dataset, concatenate_datasets

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Load HF dataset
ds = load_dataset("nvidia/CantTalkAboutThis-Topic-Control-Dataset")

# Keep only the columns your app uses (plus a couple of helpful extras)
KEEP_COLS = [
    "domain",
    "scenario",
    "system_instruction",
    "conversation",
    "distractors",
    "conversation_with_distractors",
]

# Merge train + test (some domains might only be in one split)
full = concatenate_datasets([ds["train"], ds["test"]])

# Down-select to desired domains
keep_domains = {"insurance", "real estate", "travel"}
full = full.filter(lambda ex: ex.get("domain", "").strip().lower() in keep_domains)

# Export one CSV per domain
for dom in sorted(keep_domains):
    sub = full.filter(lambda ex, d=dom: ex.get("domain", "").strip().lower() == d)
    if len(sub) == 0:
        print(f"[warn] No rows for domain: {dom}")
        continue

    # Keep only needed columns (missing cols are filled with None)
    present = [c for c in KEEP_COLS if c in sub.column_names]
    df = sub.to_pandas()[present]

    # Ensure required core columns are present (even if empty)
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = None

    out_path = os.path.join(DATA_DIR, f"{dom.replace(' ', '_')}.csv")
    df.to_csv(out_path, index=False)
    print(f"[ok] Wrote {len(df):,} rows -> {out_path}")
