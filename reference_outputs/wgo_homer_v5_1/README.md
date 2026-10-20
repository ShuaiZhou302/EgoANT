# WGO-Bench HomER V5.1 Reference Outputs

These are the 25-episode predictions used for the reported EgoANT V5.1 HomER
results. WGO-Bench videos and gold annotations are not redistributed here;
obtain them from the public WGO-Bench dataset.

## Contents

- `segmentation/`: Qwen3.6-27B GEPA coarse segmentation followed by S2
  `pad=0` full-cover local refinement.
- `candidate_selector/`: the same 308 predicted segments labeled by the
  Qwen3.5-397B four-candidate selector.
- `scores/segment_f1.json`: temporal evaluation at IoU >= 0.75 with
  outer-boundary snap.
- `scores/semantic_e2e_gemini35.json`: semantic E2E evaluation using
  `gemini-3.5-flash` and the WGO/Macrodata judge rubric.

Expected micro results:

| Metric | Result |
| --- | ---: |
| Segment F1 | `0.2030848329` |
| Temporal matches / pred / gold | `79 / 308 / 470` |
| Semantic E2E F1 | `0.1542416452` |
| Semantic matches / pred / gold | `60 / 308 / 470` |

## Recorded Configuration

| Stage | Configuration |
| --- | --- |
| Coarse segmentation | Qwen3.6-27B; full-episode contact sheets; 0.5 s sampling; 20 tiles/sheet; 224 px tiles; five columns; temperature 0; thinking disabled |
| S2 refinement | Qwen3.6-27B; `refine_full_cover_v1`; `pad_sec=0`; `min_refine_sec=3`; otherwise the same contact-sheet settings |
| Label candidates | Qwen3.5-397B; raw five-frame, ffmpeg raw five-frame, segmentation-prior neighbor, and raw-prior neighbor candidates; temperature 0; thinking disabled |
| Candidate selector | Qwen3.5-397B; five-frame PREV/CURRENT/NEXT sheets; maximum image side 1120; temperature 0; thinking disabled |
| Semantic judge | `gemini-3.5-flash`; text-only WGO/Macrodata label-judge rubric; temperature 0 |

`pad_sec=0` means the S2 request interval is not expanded and predictions are
clipped to the coarse interval. The implementation reuses precomputed
20-frame sheets that overlap that interval, so an edge sheet may contain a few
neighboring timestamps outside it. This behavior is preserved by the released
reproduction code.

The reproduction script creates a separate episode-input directory containing
only `id` and `instruction`. Gold boundaries and gold labels are not available
to coarse segmentation, S2 refinement, candidate generation, or selection;
the original gold annotations are read only during final evaluation.

The candidate-selector JSON retains all four candidate labels and the selected
source for auditability. API endpoints, credentials, machine paths, and full
model responses are not included.

## Offline Temporal Verification

After materializing WGO-Bench HomER under `/path/to/WGO_Bench`:

```bash
python scripts/wgo_bench/score_segmentation.py \
  --gold-dir /path/to/WGO_Bench/annotations/homer \
  --pred-dir reference_outputs/wgo_homer_v5_1/segmentation \
  --out /tmp/egoant_homer_segment_f1.json
```

The generated `micro` object should match `scores/segment_f1.json`. Semantic
scores require an LLM judge and may vary if the model, rubric, or provider
differs from the recorded configuration.
