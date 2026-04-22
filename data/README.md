# data/

- `raw/frfr_category/` — the Manning et al. (2023) category-condition dataset,
  reformatted to the MS-TCM canonical layout (Parquet + CSV + manifest.json).
  30 participants × 16 lists × 16 words; 4 semantic categories per list (K=4).
  Source: https://github.com/ContextLab/FRFR-analyses (PsyArXiv:
  https://psyarxiv.com/erzfp, exp2.egg).
  See [scripts/reformat_frfr_category.py](../scripts/reformat_frfr_category.py)
  to regenerate from the upstream egg.
- `processed/` — derived artifacts (gitignored; regenerate with the commands
  below). Contains `fits/` (MLE + bootstrap output from `ms-tcm fit`) and
  `embeddings/` (sentence-transformer vectors from
  `code/scripts/compute_embeddings.py`).

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

## Provenance note: 0-indexed `temporal` field

The upstream `exp2.egg` stores each recall event's source position in a
`temporal` HDF5 attribute that is **0-indexed against the `i{k}` subgroup
names in the presented group** — i.e. `temporal == 0` is the first word
presented, `temporal == 15` is the last word of a 16-word list. The
reformat script maps this to our 1-indexed `serial_position` column via
`sp = temporal + 1` (in-range values) or `sp = 0` (extra-list intrusions;
upstream `temporal` outside `[0, 15]`). This convention is verified by
cross-referencing recalled items against the presented table word-by-word.

A regression test at
[`code/tests/test_frfr.py::test_recall_serial_position_shape`](../code/tests/test_frfr.py)
pins the expected primacy + recency pattern so any future off-by-one in
the reformat would fail CI.
