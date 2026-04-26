# Clustering Analysis: FRFR-Category Behavioral Data
**Date**: 2026-04-26  
**Data**: Manning et al. 2023, FRFR-category dataset (`data/raw/frfr_category/`)  
**Goal**: Test whether the lag-CRP peakedness difference between early (category-sorted) and late (random-order) lists is consistent with the "dual alignment" hypothesis — that in early lists, temporal and semantic clustering pull toward the same adjacent items, while in late lists they pull in different directions.

---

## Data cleaning

- **Intrusions** (serial_position == 0): 10 dropped  
- **Within-list repeats**: 431 dropped (keeping first occurrence by output_position)  
- Clean recalls: 4,774 across 30 participants × 16 lists  
- Early lists = lists 0–7 (category-sorted presentation); Late = lists 8–15 (random order)

---

## Methods

### Temporal Clustering Score (TCS)
At each valid transition `prev_sp → sp`, compute the actual lag `|sp - prev_sp|` and rank it among the lags to all not-yet-recalled positions (excluding `prev_sp`). Percentile rank = (number of possible lags strictly smaller than actual) / (n_possible − 1). TCS = 1 − mean percentile rank. TCS = 1.0 means every transition was to the nearest available item; TCS = 0.5 is chance. Transitions where only 1 candidate exists are excluded (uninformative). Averaged per participant across 8 lists per condition.

### Semantic Clustering Score (SCS)
Modified Polyn et al. (2009) approach. At each transition `prev → current`:
- `actual_same` = 1 if same category, else 0  
- `expected_same` = (count of not-yet-recalled items in same category as prev) / (total not-yet-recalled, excluding prev)  
- SCS = mean(`actual_same − expected_same`) across transitions  
Positive = above-chance within-category clustering; 0 = chance. Averaged per participant.

### Co-occurrence fraction
Fraction of transitions that are BOTH |lag| = 1 AND same-category. Summed per participant across all lists in each condition, then compared.

### Lag-CRP split
Lag-CRP computed separately for within-category transitions (prev and current share category) vs. across-category transitions, using the standard Kahana (1996) denominator but partitioned by the category membership of the actual transition.

---

## Results

### 1. Temporal Clustering Score

| Condition | Mean TCS | SD |
|-|-|-|
| Early (sorted) | 0.743 | 0.065 |
| Late (random) | 0.602 | 0.086 |

- Difference (early − late): M = 0.141, 95% CI [0.101, 0.180]  
- Cohen's d (paired) = 1.334, t(29) = 7.307, p = 4.8 × 10⁻⁸  

**Interpretation**: Temporal clustering is substantially stronger in early lists. In sorted lists, the category-block structure means temporally adjacent items are also semantically coherent — so episodic/temporal context cues retrieved items presented nearby, which happen to be same-category neighbors.

### 2. Semantic Clustering Score

| Condition | Mean SCS | SD |
|-|-|-|
| Early (sorted) | 0.373 | 0.079 |
| Late (random) | 0.197 | 0.110 |

- Difference: M = 0.176, 95% CI [0.139, 0.213]  
- Cohen's d = 1.766, t(29) = 9.672, p = 1.4 × 10⁻¹⁰  

**Interpretation**: Semantic (category) clustering is also stronger in early lists. This seems paradoxical at first — why would random-order lists show *less* semantic clustering than sorted lists? The answer is that in sorted lists, temporal context is already aligned with category, so the two clustering mechanisms reinforce each other. In random lists, temporal context cues direct recall away from semantically related items (which are now scattered across serial positions), so semantic clustering competes against strong temporal pull.

### 3. Fraction of within-category transitions

| Condition | Mean | SD |
|-|-|-|
| Early | 0.566 | 0.097 |
| Late | 0.382 | 0.122 |

- Difference: M = 0.184, 95% CI [0.147, 0.221], Cohen's d = 1.869, t(29) = 10.237, p = 3.9 × 10⁻¹¹

### 4. Co-occurrence of |lag| = 1 AND same-category transitions

| Condition | Mean fraction | SD |
|-|-|-|
| Early | 0.382 | 0.107 |
| Late | 0.102 | 0.033 |

- Difference: M = 0.280, 95% CI [0.238, 0.323]  
- Cohen's d = 2.462, t(29) = 13.483, p = 5.1 × 10⁻¹⁴  

**Key finding**: In early lists, 38% of all recall transitions are simultaneously to a temporally adjacent (|lag|=1) AND same-category item. In late lists, only 10% are. Moreover, **67.6% of within-category transitions in early lists are also |lag|=1**, vs. only 26.8% in late lists. This directly confirms that temporal and semantic cues are structurally aligned in sorted lists but decoupled in random lists.

### 5. Lag-CRP decomposition

**Overall lag-CRP at lag +1**:  
- Early: 0.467, Late: 0.275

**Within-category transitions only at lag +1**:  
- Early: 0.550, Late: 0.544  ← *nearly identical*

**Across-category transitions only at lag +1**:  
- Early: 0.233, Late: 0.208  ← *nearly identical*

This is the most informative result. **The lag-CRP peakedness difference between early and late lists is NOT driven by a difference in temporal clustering within or across categories separately.** When conditioned on transition type, early and late lists show almost identical within-category CRP curves and nearly identical across-category CRP curves.

The difference in the *overall* lag-CRP is entirely explained by **composition effects**: in early lists 57% of transitions are within-category (which have high lag +1 probability ~0.55), while in late lists only 38% are. The overall lag-CRP is a mixture of two components, and early lists have a much larger weight on the high-CRP within-category component.

At lag −1, the pattern reverses oddly: within-category CRP at lag −1 is *slightly higher* in late lists (0.333 vs 0.318), and across-category is also slightly higher in late (0.102 vs 0.072). This backward-association asymmetry is consistent with late lists having more "scatter" in recall order.

TCS × SCS correlation per participant:
- Early: r = 0.724, p < 0.001 (strong positive — participants who cluster temporally also cluster semantically, because cues align)  
- Late: r = −0.534, p = 0.002 (negative — participants who cluster temporally do *less* semantic clustering, consistent with temporal and semantic cues competing)

---

## Interpretation and verdict on the hypothesis

The "dual alignment" hypothesis is **confirmed**, but the mechanism is more specific than stated:

1. In early (sorted) lists, temporal and semantic clustering are positively correlated (r = 0.72), and 68% of within-category transitions are also temporally adjacent. This is a direct structural consequence of category-blocked presentation.

2. The lag-CRP peakedness difference is **entirely a composition effect**: the within-category and across-category lag-CRP shapes are nearly the same across conditions, but early lists have a far higher fraction of within-category transitions (which have a high lag +1 peak at ~0.55) vs. across-category transitions (which have a low lag +1 peak at ~0.21).

3. In late lists, TCS and SCS are negatively correlated (r = −0.53), meaning temporal and semantic clustering actively compete: participants who follow temporal context (recall nearby items) are less likely to stay within category, and vice versa.

4. This has direct implications for the MS-TCM model: **the model needs to capture the composition effect, not just the marginal CRP shape**. A model that gets the within-category CRP right but mis-predicts the fraction of within-category transitions will get the overall lag-CRP peak wrong by construction.

---

## Limitations

- Early vs. late list assignment is confounded with list order (early always comes first in the session). Practice effects, fatigue, and strategy shifts could contribute to the difference.
- Category-sorted lists also have different inter-item similarity structure at a within-list level; we cannot fully separate the effect of category blocking from other list-level factors (e.g., list context).
- The SCS here uses the Polyn et al. 2009 "difference from chance" formula rather than the ACT-based approach; the sign and direction are robust but the absolute scale depends on this choice.
- Repeats (431 dropped) were substantial — if repeats cluster differentially in early vs. late lists, this could slightly bias the clustering scores.

---

## Figures

- `notes/clustering_analysis_figures/fig1_clustering_scores.png` — Bar plots (±SE) + individual data points for TCS, SCS, and co-occurrence fraction, early vs. late  
- `notes/clustering_analysis_figures/fig2_lagcrp_split.png` — Lag-CRP curves (±8 lags), overall and split by within/across-category transitions, early vs. late  
- `notes/clustering_analysis_figures/fig3_scatter.png` — Per-participant scatter: TCS vs SCS, and TCS vs co-occurrence fraction

---

## Code

All analysis run in Python REPL session `frfr-clustering-analysis` with `PYTHONPATH=code`, using `ms_tcm.frfr.load_frfr_category`. Key functions:

- `tcs_for_sequence(seq)` — percentile-rank TCS per recall sequence  
- `scs_for_sequence(seq, cat_seq, presented_cats)` — Polyn-style SCS per recall sequence  
- `cooccurrence_for_sequence(seq, cat_seq)` — counts |lag|=1 AND same-cat transitions  
- `crp_split(rec_df, W=16)` — lag-CRP partitioned by within/across-category  
