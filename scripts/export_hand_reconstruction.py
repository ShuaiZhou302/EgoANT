#!/usr/bin/env python3
"""Export hand reconstruction for one video.

This is a thin wrapper around the pinned VITRA reconstruction backend. It converts
backend output into the lightweight `hand_recon.json` format consumed by
`annotate_video.py`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _name):
        setattr(np, _name, _value)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_frames(video_path: Path, max_frames: int | None = None) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def project_translation(transl: np.ndarray, width: int, height: int, fov_x: float | None) -> np.ndarray:
    xy = np.full((len(transl), 2), np.nan, dtype=float)
    if fov_x is None or not np.isfinite(fov_x):
        return xy
    focal = 0.5 * width / np.tan(0.5 * float(fov_x) * np.pi / 180.0)
    z = np.clip(transl[:, 2], 1e-6, None)
    xy[:, 0] = focal * (transl[:, 0] / z) + width / 2.0
    xy[:, 1] = focal * (transl[:, 1] / z) + height / 2.0
    return xy


def convert_recon(recon: dict[str, Any], nframes: int, width: int, height: int) -> dict[str, Any]:
    fov_x = recon.get("fov_x")
    out = {
        "source_type": "hand_reconstructor",
        "camera": {"fov_x": None if fov_x is None else float(fov_x)},
        "hands": {},
    }
    for hand in ("left", "right"):
        valid = np.zeros(nframes, dtype=bool)
        wrist_world = np.full((nframes, 3), np.nan, dtype=float)
        for frame_idx, frame_result in recon.get(hand, {}).items():
            idx = int(frame_idx)
            if 0 <= idx < nframes and "transl" in frame_result:
                valid[idx] = True
                wrist_world[idx] = np.asarray(frame_result["transl"], dtype=float)[:3]
        wrist_xy = project_translation(wrist_world, width, height, fov_x)
        out["hands"][hand] = {
            "valid": valid.tolist(),
            "wrist_world": wrist_world.tolist(),
            "wrist_xy": wrist_xy.tolist(),
        }
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hand reconstruction for one video.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reconstruction-root",
        type=Path,
        default=Path(os.environ.get("HAND_RECONSTRUCTION_ROOT", "vendor/hand_reconstruction")),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("HAND_RECONSTRUCTION_MODEL_ROOT", "models/hand_reconstruction")),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fov-x", type=float, default=None, help="Reuse a known FoV and skip per-frame MoGe FoV inference.")
    parser.add_argument("--hawor-model-path", type=Path, default=None)
    parser.add_argument("--moge-model-path", type=Path, default=None)
    parser.add_argument("--detector-path", type=Path, default=None)
    parser.add_argument("--mano-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reconstruction_root = args.reconstruction_root.resolve()
    model_root = args.model_root.resolve()
    if not reconstruction_root.exists():
        raise FileNotFoundError(f"Hand-reconstruction source root not found: {reconstruction_root}")
    video_path = args.video.resolve()
    output_path = args.output.resolve()
    sys.path.insert(0, str(reconstruction_root))
    os.chdir(reconstruction_root)

    try:
        import torch
        from data.tools import hand_recon_core
    except Exception as exc:
        raise RuntimeError(
            "Cannot import the hand-reconstruction backend. Install its dependencies first."
        ) from exc

    if args.fov_x is not None:
        class FixedFovPipeline:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                pass

            def infer(self, _img: Any) -> float:
                return float(args.fov_x)

        hand_recon_core.MogePipeline = FixedFovPipeline

    Config = hand_recon_core.Config
    HandReconstructor = hand_recon_core.HandReconstructor

    class Args:
        hawor_model_path = str(args.hawor_model_path or model_root / "hawor/checkpoints/hawor.ckpt")
        moge_model_path = str(args.moge_model_path or model_root / "moge/model.pt")
        detector_path = str(args.detector_path or model_root / "hawor/external/detector.pt")
        mano_path = str(args.mano_path or model_root / "mano")

    required_paths = [Args.hawor_model_path, Args.detector_path, Args.mano_path]
    if args.fov_x is None:
        required_paths.append(Args.moge_model_path)
    missing = [p for p in required_paths if not Path(p).exists()]
    mano_right = Path(Args.mano_path) / "MANO_RIGHT.pkl"
    if not mano_right.exists():
        missing.append(str(mano_right))
    if missing:
        raise FileNotFoundError("Missing hand-reconstruction model files:\n" + "\n".join(missing))

    frames = load_frames(video_path, args.max_frames)
    height, width = frames[0].shape[:2]
    reconstructor = HandReconstructor(Config(Args), device=torch.device(args.device))
    recon = reconstructor.recon(frames)
    hand_recon = convert_recon(recon, len(frames), width, height)
    json_dump(output_path, hand_recon)
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
