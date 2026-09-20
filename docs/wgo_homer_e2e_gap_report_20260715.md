# WGO HomER E2E Gap Report

Date: 2026-07-15

## Official Reference

Official WGO-Bench / Macrodata reference:

- WGO-Bench has 100 episodes and 743 gold subtask segments.
- HomER subset has 25 egocentric episodes and 470 gold segments.
- Official best full-WGO results reported by Macrodata:
  - best segmentation: 0.306 Segment F1
  - best fixed-boundary labeling: 61.0% accuracy
  - best end-to-end pipeline: 0.168 Semantic E2E F1
- Their final E2E recipe is seeded relabeling: first predict segments, then use the original predicted label as a strong prior while showing previous/current/next segment contact sheets.

Sources:

- https://macrodata.co/blog/annotating-robot-video-subtasks
- https://huggingface.co/datasets/macrodata/WGO-Bench

## Evaluation Setup

The experiments use the HomER subset of WGO-Bench exported into the repository's
standard benchmark layout. The report intentionally avoids machine-specific
absolute paths.

Evaluation metric:

- First match predicted and gold time intervals using IoU >= 0.75 with outer-boundary snap.
- Then judge the label only on temporally matched pairs.
- Semantic E2E precision = semantic_match / n_pred.
- Semantic E2E recall = semantic_match / n_gold.
- Semantic E2E F1 = 2 * precision * recall / (precision + recall).

This means E2E is not simply segmentation F1 multiplied by labeling accuracy. Extra predicted segments and missed temporal matches directly hurt the denominator.

## Key Result Table

Rows marked in this table use the Gemini-3.5-Flash re-score with the public
WGO-style rubric. Earlier internal-judge exploratory rows should not be mixed
with these values.

| Experiment | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoANT baseline one-pass | 810/470 | 61 | 42 | 0.0953 | 0.0656 |
| 27B S2 full-cover segmentation, self labels | 308/470 | 79 | 41 | 0.2031 | 0.1054 |
| 27B S2 full-cover + 27B raw inner-0.5 relabel, Gemini judge | 308/470 | 79 | 52 | 0.2031 | 0.1337 |
| 27B S2 full-cover + 27B ffmpeg raw relabel, Gemini judge | 308/470 | 79 | 56 | 0.2031 | 0.1440 |
| 27B S2 full-cover + 27B seeded-neighbor relabel, Gemini judge | 308/470 | 79 | 49 | 0.2031 | 0.1260 |
| 27B S2 full-cover + 27B raw-prior neighbor relabel, Gemini judge | 308/470 | 79 | 55 | 0.2031 | 0.1414 |
| 27B S2 full-cover + 27B candidate selector, Gemini judge | 308/470 | 79 | 53 | 0.2031 | 0.1362 |
| 27B S2 full-cover + 397B raw relabel | 308/470 | 79 | 54 | 0.2031 | 0.1388 |
| 27B S2 full-cover + 397B ffmpeg raw relabel | 308/470 | 79 | 55 | 0.2031 | 0.1414 |
| 27B S2 full-cover + 27B-prior neighbor relabel by 397B | 308/470 | 79 | 42 | 0.2031 | 0.1080 |
| 27B S2 full-cover + raw397-prior neighbor relabel by 397B | 308/470 | 79 | 56 | 0.2031 | 0.1440 |
| 27B S2 full-cover + candidate selector by 397B, Gemini judge | 308/470 | 79 | 60 | 0.2031 | 0.1542 |

Current best automatic HomER E2E run:

```text
expS2_fullcover_candidate_select397_labels
```

## Current Pipeline

The strongest current HomER pipeline is a hybrid model pipeline:

1. Temporal segmentation:
   - Sample timestamped contact sheets from the full episode.
   - Use Qwen3.6-27B with the S2 full-cover refinement prompt to propose
     subtask boundaries and draft segment labels.
   - Score segmentation with IoU >= 0.75 and outer-boundary snap.

2. Base segment labeling:
   - For each predicted segment, sample raw frames from the current segment.
   - Use Qwen3.5-397B to produce a concise action label.
   - This raw label is the most stable single-candidate label in our ablations.

3. Contextual relabel candidates:
   - Render previous/current/next segment contact sheets with up to five frames
     per segment.
   - Run two seeded relabel variants:
     - one seeded by the 27B segmentation label;
     - one seeded by the 397B raw label.
   - Also keep an ffmpeg-frame-extraction raw label variant as an implementation
     ablation.

4. Candidate selection:
   - Give Qwen3.5-397B the current contact sheets and candidate labels.
   - Prefer the stable raw label unless another candidate is clearly better for
     the current segment.
   - Output the selected label as the final subtask annotation.

This differs from the official WGO final recipe in one important way: the
official system uses the model's original segment label as the strong prior, but
our Qwen stack works better when the 397B raw label is treated as the default
prior and other labels are used as correction candidates.

## Why The Earlier E2E Looked So Bad

The very low 0.0656 number was the old EgoANT one-pass baseline:

```text
pred=810, gold=470, IoU match=61, semantic match=42
```

It over-predicted heavily, so precision collapsed. The stronger 27B segmentation reduces prediction count to 308 and increases temporal matches to 79:

```text
pred=308, gold=470, IoU match=79
```

With 397B raw relabeling, semantic matches rise to 54, giving E2E 0.1388.
With candidate selection, semantic matches rise to 60 under Gemini-3.5-Flash
judging, giving E2E 0.1542.
The gap to official full-WGO 0.168 is therefore about 0.016, not 0.06, when
compared against our current best HomER hybrid. This is not a perfect
apples-to-apples comparison because the official 0.168 is reported on full WGO,
while this table is HomER-only.

## Experiments Run On 2026-07-15

### 1. Adjacent Segment Merge

Tried exact/verb-object/short-bridge merging on the best 27B S2 full-cover segmentation.

Result: all merge variants lowered segmentation F1.

| Merge mode | pred | IoU match | Segment F1 |
| --- | ---: | ---: | ---: |
| none | 308 | 79 | 0.2031 |
| exact same label | 305 | 77 | 0.1987 |
| verb/object | 290 | 74 | 0.1947 |
| short-bridge | 263 | 69 | 0.1883 |

Conclusion: the small fragments look ugly, but simple merging removes useful boundary hits.

### 2. Official-Style Seeded Neighbor Relabel

Used 27B segmentation labels as the seed prior, showed previous/current/next contact sheets to 397B.

Result: worse than raw relabel.

```text
semantic match: 42/79
E2E F1: 0.1080
```

Failure mode: 397B often copied neighboring segment semantics or trusted a wrong seed label too much. Example failures included changing correct raw labels like "move pitcher to center of table" into wrong labels like "close pitcher lid".

### 3. Raw-Prior Neighbor Relabel

Used 397B raw single-segment label as the seed prior, then showed previous/current/next contact sheets.

Result: small improvement.

```text
semantic match: 56/79
E2E F1: 0.1440
```

This suggests 397B raw labels are a better prior for our model than the 27B segmentation labels.

### 4. Candidate Selector

Provided 397B with the current contact sheets and four candidate labels:

- A: 397B raw-frame label
- B: 397B raw-prior neighbor relabel
- C: 27B-prior neighbor relabel
- D: ffmpeg raw-frame label

Prompt told the model to prefer A unless another candidate clearly matched the current sheet better.

Result: best automatic result so far.

```text
semantic match: 60/79
E2E F1: 0.1542
```

Candidate source distribution:

```text
all predicted segments: A=257, B=20, D=18, C=11, corrected=2
temporally matched pairs: A=67, D=6, C=3, B=3
semantic matches: A=51, B=3, D=3, C=2
```

The selector mostly keeps raw labels, but recovers several cases from alternative candidates.

## Remaining Headroom

Oracle union over available candidates:

| Candidate set | semantic match upper bound | E2E upper bound |
| --- | ---: | ---: |
| raw + raw-prior | 57 | 0.1465 |
| raw + 27B-seed neighbor | 61 | 0.1568 |
| raw + raw-prior + 27B-seed neighbor | 62 | 0.1594 |
| raw + raw-prior + 27B-seed neighbor + ffmpeg raw | 64 | 0.1645 |

This is important: the information needed to approach official full-WGO E2E is often present among our candidate labels. The weak point is automatic candidate selection / correction, not only frame extraction.

## Current Interpretation

1. 27B is still the better segmentation model for HomER-style egocentric videos.
2. 27B raw remains the strongest fixed-boundary labeler, but this advantage does not automatically transfer to noisy predicted-boundary E2E.
3. Among completed 27B-only predicted-boundary paths, 27B ffmpeg raw relabel is best at 0.1440; the 27B candidate selector reaches 0.1362.
4. 397B raw remains a strong predicted-boundary labeler, and the 397B multi-candidate selector remains the highest observed automatic E2E row.
5. Official seeded relabeling does not transfer directly to 397B when seeded from 27B labels.
6. A model-specific hybrid works better: generate multiple label candidates, then let 397B choose with a raw-label bias.
7. The largest remaining gap is segmentation recall: only 79 of 470 gold segments have IoU >= 0.75 matches. Even perfect labels on current boundaries would cap E2E at 0.2031.

## Recommended Next Tests

1. Improve candidate selector prompt using the official seeded-relabelling rules more literally, but keep the raw-label bias.
2. Add a lightweight confidence rule: when candidate A and ffmpeg raw agree semantically, lock A/D and skip neighbor correction.
3. Try a second-pass boundary refinement targeted only at currently long/ambiguous segments, because segmentation recall is still the main ceiling.
4. Re-run the best candidate-selector output with Gemini judge if we need an apples-to-apples comparison with the official WGO report.
5. Try targeted selector improvements; the completed 27B selector audit did not beat the 397B selector.
