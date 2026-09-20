# EgoANT Annotation Pipeline V5.1 Report

Date: 2026-07-16

This report summarizes the WGO-Bench-driven experiments used to tune the
current EgoANT annotation pipeline. It records the tried ideas, measured scores,
metric definitions, and the current best recipe.

## Example Assets

Included local examples (the source video is not redistributed):

- [Full-frame contact sheet](examples/pipeline_report_assets/homer1_contact_sheet_84_94.jpg).
- [Hand/object crop sheet](examples/pipeline_report_assets/homer1_handcrop_sheet_84_94.jpg).
- [Sample video + gold/pred comparison](examples/wgo_homer_1_compare/README.md).

The crop sheet is an illustrative hand/object crop from `homer_1` over the same
timestamps as the contact sheet. The fixed-boundary hand-crop ablation below was
re-run with generated HaWoR `hand_recon.json` wrist tracks, so that score is a
true hand-reconstruction crop rather than the older center-proxy crop.

## Official Reference

Sources:

- Macrodata WGO-Bench blog:
  https://macrodata.co/blog/annotating-robot-video-subtasks
- WGO-Bench dataset:
  https://huggingface.co/datasets/macrodata/WGO-Bench
- Scale dense video captioning blog that motivated contact-sheet-style
  prompting:
  https://labs.scale.com/blog/path-to-large-scale-dense-video-captioning

Official WGO-Bench reference numbers reported by Macrodata:

| Track | Official reference |
| --- | ---: |
| Segmentation | `0.306` Segment F1 |
| Fixed-boundary labeling | `61.0%` accuracy |
| End-to-end | `0.168` Semantic E2E F1 |

WGO-Bench has 100 episodes and 743 gold subtask segments. The HomER subset has
25 egocentric episodes and 470 gold segments.

## Metric Definitions

### 1. Segment F1

Segment F1 evaluates temporal boundaries only.

1. A predicted segment and a gold segment match if their temporal IoU is at
   least `0.75`.
2. Matching is greedy one-to-one, so one predicted segment cannot match multiple
   gold segments.
3. We use the WGO-style outer-boundary snap: the first predicted start and last
   predicted end are snapped to the episode boundaries before scoring.
4. Micro precision, recall, and F1 are computed over all episodes:

```text
precision = temporal_matches / predicted_segments
recall    = temporal_matches / gold_segments
F1        = 2 * precision * recall / (precision + recall)
```

This metric ignores the text label. It only asks whether the model found the
right subtask time intervals.

### 2. Fixed-Boundary Annotation Accuracy

Annotation accuracy evaluates text labels with gold boundaries fixed.

1. The model is given each gold segment boundary.
2. It predicts a subtask label for that segment.
3. A judge model decides whether the predicted label is semantically equivalent
   to the gold label.
4. Accuracy is:

```text
accuracy = semantically_correct_labels / gold_segments
```

Our local annotation ablations used a Qwen3.5-397B judge. Official Macrodata
numbers use their reported official evaluation stack, so exact judge calibration
is not perfectly apples-to-apples.

### 3. Semantic E2E F1

End-to-end scoring evaluates both temporal segmentation and label semantics.

1. First match predicted and gold time intervals using the Segment F1 temporal
   matching rule.
2. Only temporally matched pairs are sent to the semantic judge.
3. Semantic precision, recall, and F1 are:

```text
semantic_precision = semantic_matches / predicted_segments
semantic_recall    = semantic_matches / gold_segments
semantic_E2E_F1    = 2 * semantic_precision * semantic_recall
                     / (semantic_precision + semantic_recall)
```

This is why E2E can be much lower than annotation accuracy. Extra predicted
segments and missed temporal matches directly hurt the denominator before label
quality even matters.

## Ideas Tested

### A. Raw Frames vs Overlay

Original EgoANT production labeling used raw segment frames plus hand/geometry
context. We tested whether visual overlays help the VLM. They did not.

| Fixed-boundary labeler | Model | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| light overlay | Qwen3.5-397B | `48.5%` |
| raw frames | Qwen3.6-27B | `55.7%` |
| overlay | Qwen3.6-27B | `50.6%` |

Conclusion: keep raw visual input as the stable labeler default.

### B. Temporal Collage / Contact Sheets For Labeling

We tried giving the labeler timestamped temporal collages or neighboring
segment sheets instead of only raw segment frames.

| Fixed-boundary labeler | Model | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| temporal collage | Qwen3.5-397B | `45.1%` |
| neighbor contact sheet | Qwen3.5-397B | `36.8%` |
| neighbor contact sheet + timestamp rerun | Qwen3.5-397B | `35.5%` |
| temporal collage | Qwen3.6-27B | `52.8%` |
| neighbor contact sheet | Qwen3.6-27B | `34.0%` |

Conclusion: contact sheets are useful for segmentation, but they are not enough
by themselves for best fixed-boundary subtask labeling.

### C. Hand Crops / Hand Collage

We tested a hand-focused crop/collage variant for fixed-boundary labeling, but
this result needs an important caveat: WGO-Bench HomER did not include
`hand_recon.json` or hand assets in that experiment. The WGO hand-collage path
therefore used a center-proxy crop / debug hand-collage approximation rather
than the production EgoANT `raw-hand-crop` path.

| Fixed-boundary labeler | Model | Accuracy |
| --- | ---: | ---: |
| raw frames | Qwen3.5-397B | `50.2%` |
| hand collage | Qwen3.5-397B | `40.6%` |
| true hand-recon crop + raw frames | Qwen3.5-397B | `51.1%` |
| raw frames | Qwen3.6-27B | `55.7%` |
| hand collage | Qwen3.6-27B | not re-scored |

The production EgoANT `raw-hand-crop` path is different: it uses projected
`joints_xy` when available, falls back to `wrist_xy`, crops around the visible
hand points, then appends a 2x3 crop grid after the raw frames. That true path
was not the WGO HomER Track-A hand-collage input.

Follow-up result: after generating HaWoR hand reconstruction for all 25 HomER
episodes, the true hand-crop path scored `240/470 = 51.06%`, slightly above the
raw-frame `236/470 = 50.21%` Qwen3.5-397B baseline. The inputs were mostly valid crops:
`411/470` segments had `5/5` crop frames, with the remainder using partial
hand-track coverage.

Conclusion: the old WGO hand-collage approximation was misleadingly weak. True
hand-reconstruction crops can help a little, especially for local manipulation
verbs, but the gain is small and it requires an expensive hand-reconstruction
prepass. Keep raw frames as the no-asset default; use true hand crops as an
auxiliary view when hand assets are already available or cheap to generate.

### D. Contact-Sheet Segmentation

For temporal segmentation, contact sheets helped more. The strongest HomER
segmentation recipe uses timestamped 20-frame sheets, sampled at 0.5s intervals
so each sheet covers roughly 10s.

HomER-only segmentation ablations:

| Segmentation condition | Model | Segment F1 | pred/gold | matches |
| --- | ---: | ---: | ---: | ---: |
| EgoANT original line | pipeline | `0.0953` | 810/470 | 61 |
| Contact sheet legacy max3 | Qwen3.5-397B | `0.0952` | 265/470 | 35 |
| Contact sheet legacy max3 | Qwen3.6-27B | `0.1278` | 281/470 | 48 |
| Whole episode + legacy | Qwen3.6-27B | `0.1230` | 148/470 | 38 |
| Whole episode + GEPA aligned | Qwen3.6-27B | `0.1369` | 202/470 | 46 |
| S1 anti-underseg | Qwen3.5-397B | `0.1556` | 558/470 | 80 |
| S2 refine full25 | Qwen3.5-397B | `0.1674` | 498/470 | 81 |
| S2 pad=0.0 + full-cover prompt | Qwen3.6-27B | `0.2031` | 308/470 | 79 |

Conclusion: Qwen3.6-27B with the S2 full-cover prompt is our best HomER
temporal segmenter so far. It wins mainly by reducing over-fragmentation while
keeping enough temporal matches.

### E. Adjacent Segment Merge

We tried post-hoc merging predicted adjacent segments. It looked cleaner, but
hurt Segment F1.

| Merge mode | pred | matches | Segment F1 |
| --- | ---: | ---: | ---: |
| none | 308 | 79 | `0.2031` |
| exact same label | 305 | 77 | `0.1987` |
| verb/object | 290 | 74 | `0.1947` |
| short bridge | 263 | 69 | `0.1883` |

Conclusion: do not merge blindly. Some visually ugly fragments are still useful
for benchmark boundary matching.

### F. Relabeling And Candidate Selection

Once S2 full-cover predicts boundaries, label quality becomes the next bottleneck.
We tried several relabeling paths on the same HomER predicted boundaries.

| HomER E2E variant | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EgoANT one-pass baseline | 810/470 | 61 | 42 | `0.0953` | `0.0656` |
| 27B S2 full-cover, self labels | 308/470 | 79 | 41 | `0.2031` | `0.1054` |
| + Qwen3.5-397B raw relabel | 308/470 | 79 | 54 | `0.2031` | `0.1388` |
| + Qwen3.5-397B ffmpeg raw relabel | 308/470 | 79 | 55 | `0.2031` | `0.1414` |
| + 27B-prior neighbor relabel | 308/470 | 79 | 42 | `0.2031` | `0.1080` |
| + raw397-prior neighbor relabel | 308/470 | 79 | 56 | `0.2031` | `0.1440` |
| + Qwen3.5-397B candidate selector | 308/470 | 79 | 60 | `0.2031` | `0.1542` |

Conclusion: the best HomER E2E pipeline is hybrid:

- Qwen3.6-27B for segmentation.
- Qwen3.5-397B for labeling and candidate selection.

The official WGO-style seeded relabeling idea did transfer partially, but not
literally. For our Qwen stack, the 397B raw label is a better default prior than
the 27B segmentation label.

### G. Universal vs Family-Specific Full WGO

We also tested whether the HomER-best recipe should be used for all WGO subsets.
It should not.

| Pipeline | Full-WGO E2E F1 | Notes |
| --- | ---: | --- |
| GEPA coarse 397B self-label | `0.1048` | previous completed full-WGO run |
| S2 refine 397B pad=0.0 self-label | `0.0699` | over-splits robot subsets |
| Universal HomER candidate-selector pipeline | `0.1098` | same recipe for HomER/Galaxea/RoboInter |
| Family-specific mix | `0.1624` | HomER candidate-selector + robot GEPA coarse |
| Official Macrodata reference | `0.1680` | public full-WGO reference |

Subset details for the universal HomER pipeline:

| Subset | pred/gold | IoU match | semantic match | Segment F1 | E2E F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| HomER | 308/470 | 79 | 59 | `0.2031` | `0.1517` |
| Galaxea | 163/123 | 25 | 12 | `0.1748` | `0.0839` |
| RoboInter | 280/150 | 26 | 11 | `0.1209` | `0.0512` |
| Full WGO | 751/743 | 130 | 82 | `0.1740` | `0.1098` |

Conclusion: keep a family-specific policy. HomER benefits from S2 full-cover
segmentation, while robot videos currently prefer the GEPA coarse segmentation
recipe. This full-WGO transfer table is historical and should not be mixed with
the later Gemini-3.5-Flash HomER-only re-score.

## Current Best Pipeline

The current best measured configuration is:

```text
video
-> timestamped contact sheets over the full episode
   - 1 frame every 0.5s
   - 20 frames per sheet
   - about 10s per sheet
-> Qwen3.6-27B S2 full-cover temporal segmentation
-> predicted segment boundaries + draft labels
-> candidate labels for each predicted segment
   A. Qwen3.5-397B raw-frame label
   B. Qwen3.5-397B raw-prior neighbor relabel
   C. Qwen3.5-397B 27B-prior neighbor relabel
   D. Qwen3.5-397B ffmpeg raw-frame label
-> Qwen3.5-397B candidate selector
   - prefer A unless another candidate clearly matches better
   - use neighbor context only as correction evidence
-> final predicted boundaries + selected subtask labels
```

For production data outside WGO, we should separate two modes:

1. Production-safe annotation default:
   - existing fixed-boundary or wrist-minima segmentation;
   - raw-frame Qwen3.5-397B subtask labeling;
   - optional true hand-recon crop view when reliable hand assets exist;
   - conservative judge/rewrite/quality gate.
2. Experimental model segmentation mode:
   - WGO-style timestamped contact sheets;
   - Qwen3.6 S2 full-cover segmentation;
   - Qwen3.5-397B candidate-selector labeling.

This separation matters because WGO model segmentation is promising but still
benchmark-driven. It should not silently replace the stable production path for
datasets that already have reliable boundaries.

## Current Interpretation

1. Segmentation is the main ceiling. On HomER, even perfect labels on current
   predicted boundaries would cap E2E at Segment F1 `0.2031`.
2. Qwen3.6-27B is better for temporal segmentation than Qwen3.5-397B in the
   HomER setting.
3. Qwen3.5-397B is still the better fixed-boundary labeler.
4. Raw frames remain the strongest no-asset labeling visual input. Overlay,
   temporal collage, neighbor sheets, and the WGO hand-collage approximation
   underperformed raw frames, while true hand-reconstruction crops slightly
   improved HomER fixed-boundary labeling (`50.9%` vs `50.2%`).
5. Candidate selection is useful because the correct label is often present in
   one candidate even when the raw label is wrong.
6. Family-specific routing is necessary for full WGO. A universal HomER-tuned
   pipeline hurts robot subsets.

## Recommended Next Experiments

1. Improve candidate selection with stricter rules for when it is allowed to
   override the raw label.
2. Use hand crops as an auxiliary view only when the full-frame hand/object
   region is small, instead of always replacing raw frames.
3. Add adaptive segmentation routing:
   - HomER-like egocentric human videos: S2 full-cover.
   - Robot videos: GEPA coarse.
4. Re-run the best family-specific full-WGO output with the official/Gemini
   judge if strict apples-to-apples comparison is required.
