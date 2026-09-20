# EgoANT Standard Dataset Input

EgoANT annotation accepts one standard dataset layout. Dataset-specific code
should only convert raw data into this layout; the annotation pipeline should
not depend on raw dataset internals.

```text
<dataset_root>/
  meta.json
  manifest.tsv
  video/
    <task_name>/
      <id>.mp4
  hand_asset/
    <task_name>/
      <id>.npz
      <id>.meta.json
  raw_annotation/
    <task_name>/
      <id>.json
  annotation/
    <task_name>/
      <id>.json
```

`video/` is required. `hand_asset/` and `raw_annotation/` are optional standard
inputs.

If `hand_asset/<task>/<id>.npz` exists, the pipeline uses it instead of running
hand reconstruction. If it does not exist, the pipeline runs reconstruction from
the video.

If `raw_annotation/<task>/<id>.json` exists, the pipeline uses its episode-level
instruction in the final output. If it does not exist, the pipeline generates an
episode-level instruction from the final segment annotations.

## Hand Asset NPZ

The hand asset is a geometry asset, not a language annotation. It must describe
the same frame sequence as the matching video.

Required fields:

```text
camera_intrinsic             (3, 3) float32
camera_transform             (T, 4, 4) float32

left_valid                   (T,) bool
left_confidence              (T,) float32
left_hand_transform          (T, 4, 4) float32
left_finger_transforms       (T, J_left, 4, 4) float32
left_finger_names            (J_left,) object/string

right_valid                  (T,) bool
right_confidence             (T,) float32
right_hand_transform         (T, 4, 4) float32
right_finger_transforms      (T, J_right, 4, 4) float32
right_finger_names           (J_right,) object/string
```

Meaning:

- `camera_intrinsic` is the projection matrix `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`.
- `camera_transform` is the per-frame camera pose. The converter can infer
  whether it is camera-to-world or world-to-camera by checking which convention
  projects more hand points into the image.
- `left/right_hand_transform[..., :3, 3]` is the wrist/root or palm-root
  position used for speed segmentation.
- `left/right_finger_transforms[..., :3, 3]` stores finger joint positions used
  for future overlay or training.
- `valid=false` means this frame should be masked out for speed, overlay, and
  training.

The current annotation pipeline converts this asset into:

```text
hand_recon.json
  hands.left.valid
  hands.left.wrist_world
  hands.left.wrist_xy
  hands.right.valid
  hands.right.wrist_world
  hands.right.wrist_xy
```

`wrist_xy` is projected from the hand transform using camera parameters. The
caption and merge stages continue to consume the same overlay frame format.

## Hand Asset Metadata

Each `.npz` should have a matching `.meta.json`:

```json
{
  "video": "video/task_name/0.mp4",
  "pose_file": "hand_asset/task_name/0.npz",
  "source_video": "/raw/path/0.mp4",
  "source_hdf5": "/raw/path/0.hdf5",
  "coordinate_frame": "shared_with_camera_transform",
  "camera_transform_convention": "unknown_auto_project",
  "confidence_threshold": 0.2
}
```

`fps` and `nframes` are not repeated here; the pipeline reads them from the
video and stores them in the final annotation JSON.

## Raw Episode Annotation

`raw_annotation/<task>/<id>.json` is the place for dataset-provided
episode-level semantic labels. It is intentionally separate from `hand_asset`.

Minimum useful form:

```json
{
  "episode_annotation": {
    "instruction": "pour bimanual"
  },
  "source_dataset": "EgoDex",
  "source_video": "/raw/path/0.mp4",
  "source_hdf5": "/raw/path/0.hdf5"
}
```

The pipeline reads `episode_annotation.instruction` and places it near the top
of the final annotation JSON. It is not sent to the caption or merge prompts.

## Dataset Adapters

Adapters convert raw datasets into the standard layout.

For EgoDex:

```bash
python scripts/prepare_egodex_dataset.py \
  --egodex-root /path/to/EgoDex \
  --output-root /path/to/EgoDex_EgoANT \
  --max-videos 1
```

The adapter reads:

```text
<raw>/.../<task>/<id>.mp4
<raw>/.../<task>/<id>.hdf5
```

and writes:

```text
video/<task>/<new_id>.mp4
hand_asset/<task>/<new_id>.npz
raw_annotation/<task>/<new_id>.json
```
