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

The following single-video command is a lightweight exploratory path, not the
two-stage recipe used for the V5.1 result:

```bash
LOCAL_MODEL_API_KEY=local-vllm python scripts/model_segment_contact_sheets.py \
  --video /path/to/WGO_Bench/videos/homer/homer_1.mp4 \
  --output-dir outputs/model_seg/homer_1 \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.6-27B \
  --disable-thinking
```

Its generic diagnostic evaluator reports IoU 0.3 and boundary-tolerance
metrics:

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

## Exact V5.1 HomER Recipe

For the reported Segment F1 `0.2031`, run the complete coarse + S2 full-cover
pipeline:

```bash
export SEGMENT_BASE_URL=http://127.0.0.1:8000/v1
export SEGMENT_MODEL=Qwen/Qwen3.6-27B
export LABEL_BASE_URL=http://127.0.0.1:8001/v1
export LABEL_MODEL=Qwen/Qwen3.5-397B-A17B-FP8
bash scripts/reproduce_wgo_homer_v5_1.sh /path/to/WGO_Bench
```

The reported temporal metric has a separate entry point:

```bash
python scripts/wgo_bench/score_segmentation.py \
  --gold-dir /path/to/WGO_Bench/annotations/homer \
  --pred-dir reference_outputs/wgo_homer_v5_1/segmentation \
  --out /tmp/egoant_homer_segment_f1.json
```

It uses greedy one-to-one segment matching at IoU >= `0.75`, snaps only the
first predicted start and final predicted end to the gold episode extent, and
aggregates counts with micro precision and recall.

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
