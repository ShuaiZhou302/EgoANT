# WGO-Bench V5.1 Results

This note records the WGO-Bench HomER ablations that motivated the V5.1
recommendation.

Local experiment root used for reproduction:

```text
outputs/datasets/WGO_Bench_HomER
```

## Metric

Track B segmentation uses:

```text
Segment F1 at IoU >= 0.75
outer snap on the first predicted start and last predicted end
greedy one-to-one matching by IoU
micro averaging across all episodes
```

The HomER subset has:

```text
episodes: 25
gold segments: 470
```

Scorer toy cases were checked for perfect matches, outer snap behavior,
over-segmentation, and greedy one-to-one matching.

## Recommended V5.1 Pipeline

Use a hybrid model setup:

```text
Temporal segmentation: Qwen3.6-27B S2 full-cover prompt
Subtask labeling:      Qwen3.5-397B candidate-selector labeling for E2E;
                       Qwen3.6-27B raw labeling is best for fixed-boundary Label Acc.
Semantic judge:        Gemini-3.5-Flash with WGO/Macrodata rubric
```

The shorthand "398B" sometimes used in discussion refers to the 397B
collaborator endpoint used in these logs.

## Track B: Segmentation

| Condition | Model | F1 | Precision | Recall | match/pred/gold |
|---|---:|---:|---:|---:|---:|
| EgoANT original line | pipeline | 0.0953 | - | - | 61/810/470 |
| Contact sheet legacy max3 | 397B | 0.0952 | 0.1321 | 0.0745 | 35/265/470 |
| Contact sheet legacy max3 | 27B | 0.1278 | 0.1708 | 0.1021 | 48/281/470 |
| Whole episode + legacy | 27B | 0.1230 | 0.2568 | 0.0809 | 38/148/470 |
| Whole episode + GEPA aligned | 27B | 0.1369 | 0.2277 | 0.0979 | 46/202/470 |
| S1 anti-underseg | 397B | 0.1556 | 0.1434 | 0.1702 | 80/558/470 |
| S2 refine pad=1.0 | 397B | 0.1674 | 0.1627 | 0.1723 | 81/498/470 |
| S1 anti-underseg | 27B | 0.1292 | 0.1049 | 0.1681 | 79/753/470 |
| S2 refine pad=1.0 | 27B | 0.1485 | 0.1259 | 0.1809 | 85/675/470 |
| S2 refine pad=0.5 | 27B | 0.1444 | 0.1266 | 0.1681 | 79/624/470 |
| S2 refine pad=0.0 | 27B | 0.1711 | 0.1546 | 0.1915 | 90/582/470 |
| S2 refine pad=2.0 | 27B | 0.1436 | 0.1220 | 0.1745 | 82/672/470 |
| S2 pad=0.0 + midpoint full-cover postprocess | 27B | 0.1635 | 0.1478 | 0.1830 | 86/582/470 |
| S2 pad=0.0 + full-cover local prompt | 27B | 0.2031 | 0.2565 | 0.1681 | 79/308/470 |

The current best is `S2 pad=0.0 + full-cover local prompt` with Qwen3.6-27B.

## Track A: Fixed-Boundary Labeling

| Condition | Model | Accuracy | match/n |
|---|---:|---:|---:|
| raw | 27B | 55.7% | 262/470 |
| temporal collage | 27B | 52.8% | 248/470 |
| HaWoR wrist-guided crop | 27B | 52.3% | 246/470 |
| overlay | 27B | 50.6% | 238/470 |
| HaWoR wrist-guided crop | 397B | 50.9% | 239/470 |
| raw | 397B | 50.2% | 236/470 |
| overlay | 397B | 48.5% | 228/470 |
| temporal collage | 397B | 45.1% | 212/470 |
| neighbor contact sheet + timestamp rerun | 397B | 40.0% | 188/470 |
| neighbor contact sheet | 397B | 39.6% | 186/470 |
| hand collage | 397B | 39.1% | 184/470 |

The current best fixed-boundary labeler is 27B raw-frame captioning under the
Gemini-3.5-Flash judge. This does not imply the best E2E system is all-27B:
the best predicted-boundary E2E result still comes from S2 boundaries plus the
397B multi-candidate selector. The completed 27B-only predicted-boundary audit
finds 27B ffmpeg-raw relabel at 0.1440 E2E and 27B selector at 0.1362 E2E,
both below the 397B selector row.

## Qualitative Checks

Plain S2 pad=0.0 improved over the aligned baseline, but it still over-split
continuous tasks and often emitted non-completion labels such as:

```text
inspect
positioning
adjusting
rearrange
```

The full-cover prompt reduced this failure mode by asking the model to cover
the local window while preserving task-level granularity. Its improvement came
from fewer predictions and higher precision rather than from recall hacking:

```text
plain S2 pad=0.0:       pred=582, precision=0.1546, recall=0.1915
full-cover prompt S2:   pred=308, precision=0.2565, recall=0.1681
```

## Open Items

- The full-cover prompt is available as `--refine-prompt
  refine_full_cover_v1` in `scripts/wgo_bench/seg_opt_ideas.py`.
- Evaluate whether the full-cover segmentation can replace or only propose
  candidate intervals for the production wrist-minima segmenter.
- Keep raw-frame captioning as the production-safe label input. Use the
  multi-candidate selector only for the WGO-style experimental path unless it is
  validated on the target deployment dataset.
- Add a qualitative review set with predicted clips for the best and worst
  episodes, not only aggregate F1.
