"""Convert Kahana et al. 2002 Exp 1 free-recall data to our Dataset schema.

Input: data/raw/kahana2002/KahaEtal02/KahaWingJEP.exp1yng.dat
  Each trial is 5 lines separated by blank lines:
    1. subject trial cumulative_trial cond
    2. noun-pool indices of the 10 presented words (list order)
    3. serial positions of responses (1-based; -99 = not presented; -n = prior list)
    4. noun-pool indices of responses (481 = extra-list intrusion)
    5. inter-response times (ms) [dropped]

Output: data/raw/kahana2002/{presented,recalled}.parquet + manifest.json

Schema: matches ms_tcm.dataset.Dataset (used for FRFR-category). Each
participant is renumbered 0..N-1. Kahana 2002 Exp 1 uses a single storyline
per list (no category structure), so category is set to 'single'.

Usage:
    python code/scripts/convert_kahana2002.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "data/raw/kahana2002/KahaEtal02/KahaWingJEP.exp1yng.dat"
POOL = REPO / "data/raw/kahana2002/KahaEtal02/KahaWingJEP.pool"
DEST = REPO / "data/raw/kahana2002"


def _parse_trial(block: str) -> dict:
    lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
    if len(lines) < 4:
        return None  # malformed
    header = lines[0].split()
    subj = int(header[0])
    # cumulative_trial is 1-based across all experiment trials for this subject
    cum_trial = int(header[2])
    presented = [int(x) for x in lines[1].split()]
    # Recall lines: some trials may have empty response list (just line 3 blank)
    response_sps = [int(x) for x in lines[2].split()] if len(lines) > 2 else []
    response_items = [int(x) for x in lines[3].split()] if len(lines) > 3 else []
    return {
        "subject": subj,
        "cumulative_trial": cum_trial,
        "presented": presented,
        "response_sps": response_sps,
        "response_items": response_items,
    }


def main() -> None:
    text = SRC.read_text()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    print(f"parsed {len(blocks)} blocks")

    # Load word pool (1-based noun indices).
    pool = [w.strip() for w in POOL.read_text().splitlines() if w.strip()]
    print(f"word pool size: {len(pool)}")

    # Parse all trials.
    trials = [t for t in (_parse_trial(b) for b in blocks) if t is not None]
    print(f"parsed {len(trials)} trials")

    # Drop practice trials: C&Z excludes the first 3 of 33 per participant.
    # Kahana 2002 README confirms lists 1-3 were practice.
    trials = [t for t in trials if t["cumulative_trial"] > 3]
    print(f"after dropping practice (cumulative_trial<=3): {len(trials)} trials")

    # Renumber participants 0..N-1 (deterministic ordering by subject ID).
    subjects = sorted(set(t["subject"] for t in trials))
    subj_to_part = {s: i for i, s in enumerate(subjects)}
    print(f"n participants: {len(subjects)}")

    # Renumber list 0..M-1 per participant (30 lists after dropping 3 practice).
    pres_rows = []
    rec_rows = []
    for t in trials:
        part = subj_to_part[t["subject"]]
        # After dropping practice, list index = cumulative_trial - 4 (so first
        # analyzable list becomes list 0).
        lst = t["cumulative_trial"] - 4
        # Presented.
        for sp, pool_idx in enumerate(t["presented"], start=1):
            # Word pool indices are 1-based. pool[] is 0-indexed.
            word = pool[pool_idx - 1] if 1 <= pool_idx <= len(pool) else f"W{pool_idx}"
            pres_rows.append({
                "participant": part,
                "list": lst,
                "serial_position": sp,
                "word": word,
                "category": "single",  # no category structure in Kahana 2002
                "size": "small", "first_letter": word[0].upper() if word else "X",
                "word_length": len(word),
                "color_r": 0, "color_g": 0, "color_b": 0,
                "pos_x": 0.0, "pos_y": 0.0,
                "list_group": "early" if lst < 15 else "late",
            })

        # Recalls.
        for out_pos, (sp, pool_idx) in enumerate(
            zip(t["response_sps"], t["response_items"]), start=1,
        ):
            # SP encoding: positive = serial position in THIS list; -99 = not
            # presented; negative (other) = prior-list intrusion; 481 = extra-list.
            # For single-trial free recall we collapse all "not in this list" cases
            # to sp=0 (extra-list intrusion by our schema convention).
            if sp > 0:
                this_sp = int(sp)
                word = pool[pool_idx - 1] if 1 <= pool_idx <= len(pool) else f"W{pool_idx}"
            else:
                this_sp = 0  # intrusion
                word = pool[pool_idx - 1] if 1 <= pool_idx <= len(pool) else "XLI"
            rec_rows.append({
                "participant": part,
                "list": lst,
                "output_position": out_pos,
                "word": word,
                "category": "single",
                "serial_position": this_sp,
                "list_group": "early" if lst < 15 else "late",
            })

    pres_df = pd.DataFrame(pres_rows)
    rec_df = pd.DataFrame(rec_rows)
    print(f"presented rows: {len(pres_df)}, recalled rows: {len(rec_df)}")

    # Write parquet.
    DEST.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pres_df, preserve_index=False), DEST / "presented.parquet")
    pq.write_table(pa.Table.from_pandas(rec_df, preserve_index=False), DEST / "recalled.parquet")

    # Write manifest.json (matches Dataset schema expected by ms_tcm.dataset.load_dataset).
    n_parts = int(pres_df["participant"].nunique())
    n_lists = int(pres_df.groupby("participant")["list"].nunique().max())
    n_words = int(pres_df.groupby(["participant", "list"]).size().max())
    manifest = {
        "schema_version": "1.0.0",
        "dataset_name": "kahana2002_exp1_young",
        "design": {
            "participants": n_parts,
            "lists_per_participant": n_lists,
            "words_per_list": n_words,
            "unique_categories_per_list": 1,
            "unique_categories_total": 1,
        },
        "source": {
            "paper": "Kahana, Howard, Zaromb, Wingfield (2002) JEP:LMC 28(3), 530-540",
            "url": "https://memory.psych.upenn.edu/Data_Archive",
            "condition": "Experiment 1, young adults (immediate free recall)",
        },
        "preprocessing": "Dropped first 3 practice lists per participant (C&Z 2025 convention). Intrusions (-99 not-presented, prior-list, extra-list) mapped to serial_position=0.",
        "row_counts": {
            "presented": int(len(pres_df)),
            "recalled_total": int(len(rec_df)),
            "recalled_in_list": int((rec_df["serial_position"] > 0).sum()),
            "recalled_extra_list_intrusions": int((rec_df["serial_position"] == 0).sum()),
        },
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Summary.
    print(f"\nWrote {DEST}/")
    print(f"  manifest: {manifest}")
    print(f"  recalls per list: mean={rec_df.groupby(['participant','list']).size().mean():.2f}, "
          f"median={rec_df.groupby(['participant','list']).size().median():.0f}")
    print(f"  intrusion rate: {(rec_df['serial_position']==0).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
