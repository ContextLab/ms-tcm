# data/

- `raw/frfr_category/` — the Manning et al. (2023) category-condition dataset,
  reformatted to the MS-TCM canonical layout (Parquet + CSV + manifest.json).
  30 participants × 16 lists × 16 words; 4 semantic categories per list (K=4).
  Source: https://github.com/ContextLab/FRFR-analyses (exp2.egg). See
  [scripts/reformat_frfr_category.py](../scripts/reformat_frfr_category.py)
  to regenerate from the upstream egg.
- `processed/` — derived artifacts. Fit outputs go under `processed/fits/<name>/`
  and are gitignored.

## Canonical dataset layout

Every dataset directory conforms to the schema in
[specs/001-ms-tcm-impl/contracts/dataset-schema.md](../specs/001-ms-tcm-impl/contracts/dataset-schema.md):

```
<dataset_dir>/
├── manifest.json          # provenance + per-file SHA-256 hashes
├── presented.parquet      # one row per presented word
├── recalled.parquet       # one row per recalled word
├── presented.csv          # CSV mirror (diff-able)
└── recalled.csv           # CSV mirror (diff-able)
```

## Verifying the bundled dataset

```bash
ms-tcm validate data/raw/frfr_category
```

Expected: exit 0. Any hash mismatch or schema violation is named by file /
column / row. The seven rules are listed in data-model.md §7.

## Regenerating from the source egg

```bash
python scripts/reformat_frfr_category.py      # downloads exp2.egg + reformats
```

Running the script twice with the same input egg produces byte-identical
output (SHA-256 hashes in manifest.json unchanged across runs).
