# Contract: Dataset Schema

**Consumer**: every notebook, script, CLI command, and test that reads or writes a dataset.
**Producer**: `ms_tcm.schema`, `ms_tcm.dataset`, `ms_tcm.frfr`, `ms_tcm.io`.
**Format**: Apache Parquet binary tables plus plain-text CSV and JSON sidecars, co-located in one dataset directory.

## 1. Directory layout

```text
<dataset_dir>/
├── manifest.json          # plain text; provenance + per-file SHA-256 hashes
├── presented.parquet      # one row per presented word
├── recalled.parquet       # one row per recalled word (intrusions included; flag = serial_position 0)
├── presented.csv          # CSV mirror of presented.parquet (diff-able)
└── recalled.csv           # CSV mirror of recalled.parquet (diff-able)
```

Both Parquet files are required. Both CSV mirrors are required (committed for version-control diff-ability per FR-010). `manifest.json` is required and is written last so its hashes cover every other file.

There is no separate `storylines.parquet`: storyline identity is the `category` column of `presented.parquet` (K = 4 unique categories per list). There is no `cues.parquet` or `references.parquet`: the free-recall paradigm used in this feature does not emit cue–target pairs or conversational references.

## 2. Parquet write conventions (all files)

| Setting | Value | Reason |
|-|-|-|
| `compression` | `"zstd"` | Cross-platform deterministic at level 1 |
| `compression_level` | `1` | Lowest level that is bit-exact across platforms |
| `use_dictionary` | `False` | Dictionary order depends on insertion order |
| `write_statistics` | `False` | Row-group statistics are float-reduction-sensitive |
| `version` | `"2.6"` | Stable, pinned |
| Row ordering | Sorted by primary key before write | Eliminates write-time ordering nondeterminism |

CSV mirrors are written by pandas with default (deterministic) options and no index column.

## 3. Table schemas

### `presented.parquet`

```text
participant:      int64  not null
list:             int64  not null
serial_position:  int64  not null      # 1-based within (participant, list)
word:             string not null
category:         string not null
size:             string not null      # "small" | "large"
first_letter:     string not null
word_length:      int64  not null
color_r:          int64  not null      # 0..254, or -1 when reduced
color_g:          int64  not null
color_b:          int64  not null
pos_x:            double nullable      # NaN when reduced
pos_y:            double nullable
list_group:       string not null      # "early" | "late"
```

Sorted ascending by `(participant, list, serial_position)`. Primary key: `(participant, list, serial_position)`.

### `recalled.parquet`

```text
participant:      int64  not null
list:             int64  not null
output_position:  int64  not null      # 1-based within (participant, list) recall output
word:             string not null
category:         string not null      # may be "" for extra-list intrusions
serial_position:  int64  not null      # 1..W if from this list; 0 if extra-list intrusion
list_group:       string not null
```

Sorted ascending by `(participant, list, output_position)`. Primary key: `(participant, list, output_position)`.

## 4. `manifest.json`

The required top-level keys (enumerated by `ms_tcm.schema.MANIFEST_KEYS`) and their content:

```json
{
  "source": {
    "paper": "Manning et al. 2023 (https://github.com/ContextLab/FRFR-analyses)",
    "condition": "category (exp2)",
    "egg_url": "https://www.dropbox.com/s/kliq92lta7mvqcc/exp2.egg?dl=1",
    "egg_sha256": "<hex>"
  },
  "design": {
    "participants": 30,
    "lists_per_participant": 16,
    "words_per_list": 16,
    "unique_categories_per_list": 4,
    "unique_categories_total": 15,
    "early_lists": "list < 8 (sorted by category)",
    "late_lists": "list >= 8 (random order)"
  },
  "files": {
    "presented.parquet": { "sha256": "<hex>", "size_bytes": <int> },
    "recalled.parquet":  { "sha256": "<hex>", "size_bytes": <int> },
    "presented.csv":     { "sha256": "<hex>", "size_bytes": <int> },
    "recalled.csv":      { "sha256": "<hex>", "size_bytes": <int> }
  },
  "row_counts": {
    "presented": 7680,
    "recalled_total": 5215,
    "recalled_in_list": 4822,
    "recalled_extra_list_intrusions": 393
  },
  "created_at": "2026-04-21T00:00:00+00:00"
}
```

Written with sorted keys, two-space indentation, and a trailing newline (settings frozen in `ms_tcm.io.write_json_sorted`). Validation MUST re-hash every file in `files` and raise on any mismatch (rule 6).

## 5. Validation rules

The `validate_dataset(path)` routine enumerates exactly the seven rules documented in `data-model.md` §7. Each violation is reported by file, column, and offending row.

## 6. Schema versioning

- This contract is **1.0.0** (free-recall schema with `presented` + `recalled` tables).
- **MAJOR**: removing a required column or changing its type incompatibly; removing `presented` or `recalled`.
- **MINOR**: adding an optional column, an optional auxiliary table, or expanding an enum (e.g. a third `list_group` value).
- **PATCH**: documentation, formatting, key reordering in `manifest.json` that does not change bytes.

Old datasets MUST remain readable until a deprecation cycle has passed.

## 7. Extending the schema to other datasets

Any free-recall dataset with (a) word-level presentation rows and (b) word-level recall rows can be adapted to this schema. The only FRFR-specific columns are the visual features (`color_*`, `pos_*`); when a future dataset lacks them, the adapter writes `-1` (ints) or `NaN` (floats) sentinels and the feature encoder drops the corresponding blocks. Any adapter that produces a conformant directory can be fit by `ms-tcm fit` with no code changes.
