# EgoANT v5.1

**Egocentric video segmentation and annotation with open-weight
vision-language models.**

[Technical report](https://www.shuai-zhou.com/egoant-report/?lang=en) ·
[中文报告](https://www.shuai-zhou.com/egoant-report/?lang=zh) ·
[Release notes](docs/egoant_pipeline_v5_1_report.md) ·
[Results ledger](docs/wgo_bench_v5_1_results.md)

EgoANT turns long first-person manipulation videos into temporally aligned
atomic actions: segment boundaries plus short subtask labels. Version 5.1 is a
stable research release of the pipeline used in the accompanying report. It
contains the WGO-Bench experiment code, prompts, evaluation utilities, and the
single-video annotation pipeline; datasets, private outputs, API credentials,
and restricted model files are not included.

## v5.1 at a glance

All reported WGO numbers use the 25-video HomER subset with 470 reference
segments.

| Stage | Best v5.1 result | Configuration |
| --- | ---: | --- |
| Temporal segmentation | Segment F1 `0.2031` | Qwen3.6-27B, full-episode contact sheets, S2 local refinement |
| Fixed-boundary labeling | Label accuracy `55.7%` (`262/470`) | Qwen3.6-27B, raw frames |
| End to end | Semantic E2E F1 `0.1542` | 27B segmentation, 397B multi-candidate selector |

The main finding is that stage-specific model choice matters more than using
one large model everywhere. The 27B model is strongest for segmentation and
fixed-boundary raw-frame labeling in these experiments, while the best
predicted-boundary result still uses the 397B selector.

```text
video
  -> timestamped contact sheets
  -> 27B full-episode segmentation
  -> S2 full-cover boundary refinement
  -> multiple 397B label candidates
  -> 397B candidate selector
  -> timestamped subtask annotations
```

## Repository layout

```text
docs/                         experiment reports, schemas, and worked examples
models/providers.json         OpenAI-compatible endpoint configuration
prompts/                      maintained prompts used by the release
scripts/                      production and evaluation entry points
scripts/wgo_bench/            WGO fixed-boundary and selector utilities
tests/                        unit tests
vendor/hand_reconstruction/   pinned VITRA submodule (optional)
```

## Installation

Clone the repository and its pinned third-party dependencies:

```bash
git clone --recurse-submodules https://github.com/ShuaiZhou302/EgoANT.git
cd EgoANT
```

Create the core environment:

```bash
conda env create -f environment.yml
conda activate egoant
```

or install the same dependencies in an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Configure an OpenAI-compatible VLM endpoint in `models/providers.json`. Keep
keys in environment variables; do not write credentials into the JSON file.

```bash
export LOCAL_MODEL_API_KEY=local-vllm
# or, for a hosted provider:
export ANNOTATION_API_KEY=...
```

### Optional hand reconstruction

The WGO segmentation and raw-frame labeling paths do not require hand
reconstruction. The production wrist-trajectory path and hand-crop ablations
use VITRA/HaWoR and require additional GPU dependencies and weights.

```bash
git submodule update --init --recursive
bash scripts/setup_models.sh
```

`setup_models.sh` downloads the public HaWoR checkpoint, detector, and model
configuration. MANO is license-gated: download `MANO_RIGHT.pkl` from the
[official MANO site](https://mano.is.tue.mpg.de/) and place it at:

```text
models/hand_reconstruction/mano/MANO_RIGHT.pkl
```

The optional reconstruction stack is **not** covered by EgoANT's MIT license.
In particular, HaWoR is non-commercial and no-derivatives. Read
[THIRD_PARTY.md](THIRD_PARTY.md) before using it.

## Reproduce the WGO workflow

Materialize the HomER subset from WGO-Bench:

```bash
python scripts/prepare_wgo_bench.py \
  --dataset macrodata/WGO-Bench \
  --subset homer \
  --limit 25 \
  --output-root /path/to/WGO_Bench
```

This creates:

```text
/path/to/WGO_Bench/
  videos/homer/homer_1.mp4
  annotations/homer/homer_1.json
  manifest.json
```

Run the complete V5.1 HomER recipe against your own OpenAI-compatible model
deployments:

```bash
export SEGMENT_BASE_URL=http://127.0.0.1:8000/v1
export SEGMENT_MODEL=Qwen/Qwen3.6-27B
export LABEL_BASE_URL=http://127.0.0.1:8001/v1
export LABEL_MODEL=Qwen/Qwen3.5-397B-A17B-FP8
export LOCAL_MODEL_API_KEY=local-vllm

bash scripts/reproduce_wgo_homer_v5_1.sh \
  /path/to/WGO_Bench \
  outputs/wgo_homer_v5_1
```

This runs the actual sequence used by the reported experiment:

1. create generation inputs containing only the episode ID and instruction;
2. whole-episode, timestamped contact-sheet segmentation with the GEPA-style
   completed-event prompt;
3. S2 local refinement with `pad=0` and the full-cover prompt;
4. raw, ffmpeg-raw, segmentation-prior neighbor, and raw-prior neighbor label
   candidates;
5. the Qwen3.5-397B PREV/CURRENT/NEXT candidate selector;
6. WGO temporal scoring and, when `GEMINI_API_KEY` is set, semantic E2E
   scoring.

Gold boundaries and gold subtask labels are removed before generation. The
segmentation, labeling, and selector stages receive only video, episode ID,
episode instruction, and earlier model predictions. Gold segments are opened
again only by the final evaluation stage.

Here `pad=0` means that the declared S2 interval is not expanded and returned
boundaries are clipped to the coarse interval. S2 reuses precomputed contact
sheets at 20-tile granularity, so the first or last overlapping sheet can show
neighboring timestamps outside that interval. This is the behavior used for
the reported result.

The exact S2 implementation is
[`scripts/wgo_bench/seg_opt_ideas.py`](scripts/wgo_bench/seg_opt_ideas.py), and
the exact selector is
[`scripts/wgo_bench/select_pred_segment_candidates.py`](scripts/wgo_bench/select_pred_segment_candidates.py).
Prompts are also published separately in [`prompts/`](prompts/).

Re-evaluate the released predictions without making any model calls:

```bash
python scripts/wgo_bench/score_segmentation.py \
  --gold-dir /path/to/WGO_Bench/annotations/homer \
  --pred-dir reference_outputs/wgo_homer_v5_1/segmentation \
  --out /tmp/egoant_homer_segment_f1.json
```

Expected micro result: `79/308/470`, Segment F1 `0.2030848329`. The older
`scripts/evaluate_subtask_segments.py` is a generic diagnostic utility with a
configurable threshold; it is not the entry point used for the reported WGO
numbers.

The exact four-candidate selector loads aligned label directories directly.
For a separate assembly step, create the `candidate_labels` JSON consumed by
the generic selector with:

```bash
python scripts/wgo_bench/assemble_candidate_labels.py \
  --boundary-dir outputs/wgo_homer_v5_1/02_s2_full_cover \
  --candidate \
    raw=outputs/wgo_homer_v5_1/03_candidate_raw \
    rawprior=outputs/wgo_homer_v5_1/03_candidate_rawprior_neighbor \
    seed=outputs/wgo_homer_v5_1/03_candidate_seed_neighbor \
    ffmpeg=outputs/wgo_homer_v5_1/03_candidate_ffmpeg \
  --output-dir outputs/wgo_homer_v5_1/03_candidates_assembled
```

All 25 released predictions and both recorded score files are under
[`reference_outputs/wgo_homer_v5_1/`](reference_outputs/wgo_homer_v5_1/).

See [the v5.1 results ledger](docs/wgo_bench_v5_1_results.md) for the complete
ablation table and [the full report](docs/egoant_pipeline_v5_1_report.md) for
method details.

## Single-video pipeline

For data that already contains a compatible hand asset, the pipeline can run
without the optional reconstruction dependency:

```bash
export VIDEO=/path/to/video.mp4
export HAND_SOURCE=asset
export HAND_ASSET=/path/to/hand_asset.npz
export OUT=/path/to/output
export MODEL_PROVIDER=local_vllm
bash scripts/run_lft_pipeline.sh
```

To reconstruct hands from video instead, initialize the submodules and models,
then use `HAND_SOURCE=recon` (the default). Output is written as final JSON,
quality and token reports, and optional segment clips. `DEBUG=1` retains
intermediate frames, prompts, responses, votes, and diagnostics.

The normalized dataset layout and hand-asset schema are documented in
[docs/hand_pose_asset_schema.md](docs/hand_pose_asset_schema.md).

## Release provenance

This public repository is a clean release snapshot derived from the private
development branch `Ver_5.1` at source commit
`dd803ebbe7d50e48e3ac4b39c93c386ad1ebc95e`. Reference JSON predictions and
score artifacts for HomER are included under `reference_outputs/`; videos,
credentials, provider configuration, and private development history are
excluded. The original contributors are acknowledged in [AUTHORS.md](AUTHORS.md).

## Citation

Use the metadata in [CITATION.cff](CITATION.cff), or cite the versioned GitHub
release. The technical report is available at
<https://www.shuai-zhou.com/egoant-report/?lang=en>.

## License

EgoANT's original code, prompts, tests, and documentation are released under
the [MIT License](LICENSE). Third-party software, model weights, and datasets
retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
