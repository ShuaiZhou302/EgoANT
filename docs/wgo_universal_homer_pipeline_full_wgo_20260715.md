# Universal HomER Pipeline on Full WGO

Date: 2026-07-15

## Question

Evaluate the same HomER-best V5.1 pipeline on all three WGO-Bench subsets,
without using a family-specific strategy.

The universal pipeline is:

```text
video
-> timestamped contact sheets
-> Qwen3.6-27B S2 full-cover temporal segmentation
-> candidate labels for each predicted segment
   - Qwen3.5-397B raw-frame label
   - Qwen3.5-397B raw-prior neighbor relabel
   - Qwen3.5-397B 27B-prior neighbor relabel
   - Qwen3.5-397B ffmpeg raw-frame label
-> Qwen3.5-397B candidate selector
-> final segment boundaries + selected subtask labels
```

## What Was Run

HomER reused the existing best candidate-selector output.

Galaxea and RoboInter were newly run through the same recipe:

1. Qwen3.6-27B GEPA coarse segmentation.
2. Qwen3.6-27B S2 pad=0.0 local full-cover refinement.
3. Qwen3.5-397B raw-frame labels.
4. Qwen3.5-397B raw-prior neighbor relabel.
5. Qwen3.5-397B 27B-prior neighbor relabel.
6. Qwen3.5-397B ffmpeg raw-frame labels.
7. Qwen3.5-397B candidate selector.

Robot-side completion:

```text
Galaxea:   25/25 episodes
RoboInter: 50/50 episodes
Robot predicted segments after S2: 443
Raw labels failed: 0/443
Raw-prior relabel failed: 0/443
27B-prior relabel failed: 0/443
ffmpeg raw label failed: 9/443
Candidate selector failed: 0/443
```

The ffmpeg failures are isolated D-candidate failures only; the selector still
had raw/raw-prior/27B-prior candidates.

## Result

Semantic E2E uses IoU >= 0.75 with outer-boundary snap for temporal matching,
then judges labels only on matched segment pairs.

This file is a historical transfer check for applying the HomER-tuned pipeline
to all WGO subsets. The robot-subset semantic rows were not part of the later
Gemini-3.5-Flash HomER re-score. Use the HomER-only reports for the final
Gemini-judged EgoANT V5.1 claim.

| Subset | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HomER | 308/470 | 79 | 59 | 0.2031 | 0.1517 |
| Galaxea | 163/123 | 25 | 12 | 0.1748 | 0.0839 |
| RoboInter | 280/150 | 26 | 11 | 0.1209 | 0.0512 |
| Full WGO | 751/743 | 130 | 82 | 0.1740 | 0.1098 |

Official full-WGO reference reported by Macrodata:

```text
Semantic E2E F1: 0.168
```

Under this historical transfer-check scoring, the universal HomER pipeline is
below the official result by about `0.0582`.

## Comparison

| Pipeline | Full-WGO E2E F1 | Notes |
| --- | ---: | --- |
| GEPA coarse 397B self-label | 0.1048 | Previous full-WGO completed run |
| S2 refine 397B pad=0.0 self-label | 0.0699 | Over-splits robot subsets badly |
| Universal HomER candidate-selector pipeline | 0.1098 | This report |
| Family-specific mix | 0.1624 | HomER candidate-selector + robot GEPA coarse |
| Official Macrodata reference | 0.1680 | Public full-WGO reference |

The universal HomER pipeline is slightly above the older GEPA full-WGO baseline,
but it is much worse than the family-specific strategy.

## Candidate Selector Behavior

Across all 751 predicted segments:

```text
A raw-frame label:              610
B raw-prior neighbor relabel:    38
C 27B-prior neighbor relabel:    48
D ffmpeg raw-frame label:        53
corrected:                        2
```

By subset:

```text
HomER:     A=257, B=20, C=11, D=18, corrected=2
Galaxea:   A=118, B=16, C=16, D=13
RoboInter: A=235, B=2,  C=21, D=22
```

The selector mostly kept raw labels. The poor full-WGO score is therefore not
primarily caused by selector instability. The main issue is temporal
segmentation on robot subsets.

## Interpretation

HomER-style S2 full-cover refinement is useful for HomER but does not transfer
cleanly to robot videos.

The robot subsets degrade because:

- Galaxea: predicted segment count increases to 163 for 123 gold segments, but
  only 25 temporal matches survive IoU >= 0.75.
- RoboInter: predicted segment count increases to 280 for 150 gold segments,
  but only 26 temporal matches survive. The old GEPA coarse robot run had 56
  RoboInter temporal matches.

This means the HomER S2 recipe over-fragments or misplaces boundaries for robot
videos. Better labels cannot rescue E2E when temporal matches are lost.

## Recommendation

Do not use one universal HomER-tuned pipeline for full WGO.

Use a family-specific policy instead:

```text
HomER:     27B S2 full-cover + 397B candidate selector
Galaxea:   GEPA coarse recipe
RoboInter: GEPA coarse recipe
```

This is currently the strongest measured configuration and reaches about
`0.1624` full-WGO E2E, close to the official `0.168` reference.

If we want a single universal pipeline, the next work should focus on a routing
or adaptive segmentation rule:

1. classify video family or motion style;
2. use HomER S2 only when long egocentric human manipulation needs refinement;
3. keep GEPA coarse segmentation for short robot videos;
4. then apply the candidate-selector label stage only after the right
   segmentation recipe is chosen.
