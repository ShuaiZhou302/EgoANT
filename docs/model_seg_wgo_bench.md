# WGO-Bench model-driven segmentation

Goal: test model-driven temporal segmentation before changing the production
EgoANT boundary detector.

## WGO-Bench

Dataset:

```bash
hf download macrodata/WGO-Bench data/annotations.parquet \
  --repo-type dataset \
  --local-dir /path/to/WGO-Bench
```

The Hugging Face dataset has 100 rows with:

- `id`
- `video` bytes
- `instruction`
- `segments`: `start_sec`, `end_sec`, `subtask`
- `metadata`

Materialize the full HomER subset:

```bash
python scripts/prepare_wgo_bench.py \
  --output-root /path/to/WGO_Bench \
  --subset homer \
  --limit 25
```

This writes:

- `/path/to/WGO_Bench/videos/homer/<id>.mp4`
- `/path/to/WGO_Bench/annotations/homer/<id>.json`

## Segmentation Prompt

The experimental script follows Refiner/WGO-Bench defaults:

- sample one frame every `0.5s`
- `20` frames per contact sheet, so one sheet covers about `10s`
- `5` columns
- visible timestamp badge on each tile
- prompt asks the model to segment only completed manipulation events

Run:

```bash
LOCAL_MODEL_API_KEY=local-vllm python scripts/model_segment_contact_sheets.py \
  --video /path/to/WGO_Bench/videos/homer/homer_1.mp4 \
  --output-dir outputs/model_seg/homer_1 \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-27B \
  --disable-thinking
```

Evaluate:

```bash
python scripts/evaluate_subtask_segments.py \
  --gold /path/to/WGO_Bench/annotations/homer/homer_1.json \
  --pred outputs/model_seg/homer_1/model_segments.json \
  --output outputs/model_seg/homer_1/eval.json
```

Primary metrics:

- `boundary_f1`: end-boundary matching with `1.0s` tolerance
- `segment_f1`: predicted/gold segment matching at IoU >= `0.3`
- `mean_matched_iou`
- `mean_text_similarity`

## Next Hypotheses

1. Direct model segmentation may produce better coarse task boundaries than
   wrist minima when hand motion is noisy.
2. It may miss short contact-only events because `0.5s` sampling can skip a
   brief state change.
3. If direct segmentation is noisy, use it only to propose coarse candidate
   intervals, then run the existing EgoANT per-segment captioner and merge judge.
4. The WGO-style seeded relabeling step is likely more useful than replacing
   EgoANT captions outright: keep fixed boundaries, show previous/current/next
   contact sheets, and minimally correct the seed label.
