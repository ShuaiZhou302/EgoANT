#!/usr/bin/env python3
"""Normalize EgoDex videos and HDF5 hand/camera data for EgoANT annotation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXCLUDED_HAND_PARTS = {"Arm", "Forearm", "Shoulder"}


def clean_task_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "task"


def readable_task(value: str) -> str:
    return clean_task_name(value).replace("_", " ")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "skip":
        return
    else:
        raise ValueError(f"Unknown video mode: {mode}")


def h5_dataset(file: h5py.File, key: str, default: np.ndarray | None = None) -> np.ndarray | None:
    if key not in file:
        return default
    return np.asarray(file[key])


def collect_finger_transforms(file: h5py.File, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    transform_group = file["transforms"]
    names: list[str] = []
    arrays: list[np.ndarray] = []
    for key in sorted(transform_group.keys()):
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if not suffix or suffix in EXCLUDED_HAND_PARTS or key == f"{prefix}Hand":
            continue
        value = np.asarray(transform_group[key])
        if value.ndim == 3 and value.shape[1:] == (4, 4):
            names.append(suffix)
            arrays.append(value)

    if not arrays:
        t = np.asarray(file[f"transforms/{prefix}Hand"]).shape[0]
        return np.empty((t, 0, 4, 4), dtype=np.float32), np.asarray([], dtype=object)

    min_len = min(arr.shape[0] for arr in arrays)
    stacked = np.stack([arr[:min_len] for arr in arrays], axis=1).astype(np.float32)
    return stacked, np.asarray(names, dtype=object)


def write_hand_asset(hdf5_path: Path, output_path: Path, confidence_threshold: float) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as file:
        camera_intrinsic = h5_dataset(file, "camera/intrinsic")
        camera_transform = h5_dataset(file, "transforms/camera")
        if camera_intrinsic is None or camera_transform is None:
            raise KeyError(f"{hdf5_path} is missing camera/intrinsic or transforms/camera")

        arrays: dict[str, np.ndarray] = {
            "camera_intrinsic": camera_intrinsic.astype(np.float32),
            "camera_transform": camera_transform.astype(np.float32),
        }
        meta: dict[str, Any] = {
            "source_hdf5": str(hdf5_path),
            "coordinate_frame": "shared_with_camera_transform",
            "camera_transform_convention": "unknown_auto_project",
            "confidence_threshold": confidence_threshold,
            "hands": {},
        }

        for hand in ("left", "right"):
            prefix = "left" if hand == "left" else "right"
            hand_transform = h5_dataset(file, f"transforms/{prefix}Hand")
            if hand_transform is None:
                raise KeyError(f"{hdf5_path} is missing transforms/{prefix}Hand")
            confidence = h5_dataset(file, f"confidences/{prefix}Hand")
            if confidence is None:
                confidence = np.ones(hand_transform.shape[0], dtype=np.float32)
            confidence = confidence.astype(np.float32)
            valid = np.isfinite(hand_transform[:, :3, 3]).all(axis=1) & (confidence >= confidence_threshold)
            finger_transforms, finger_names = collect_finger_transforms(file, prefix)

            arrays[f"{hand}_valid"] = valid.astype(bool)
            arrays[f"{hand}_confidence"] = confidence
            arrays[f"{hand}_hand_transform"] = hand_transform.astype(np.float32)
            arrays[f"{hand}_finger_transforms"] = finger_transforms
            arrays[f"{hand}_finger_names"] = finger_names
            meta["hands"][hand] = {
                "hand_transform": f"{hand}_hand_transform",
                "finger_transforms": f"{hand}_finger_transforms",
                "finger_names": finger_names.tolist(),
                "confidence": f"{hand}_confidence",
                "valid": f"{hand}_valid",
            }

        np.savez_compressed(output_path, **arrays)
    return meta


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_items(root: Path) -> list[tuple[str, Path, Path]]:
    items: list[tuple[str, Path, Path]] = []
    for hdf5_path in sorted(root.glob("**/*.hdf5")):
        video_path = hdf5_path.with_suffix(".mp4")
        if not video_path.exists():
            continue
        task = clean_task_name(hdf5_path.parent.name)
        items.append((task, video_path, hdf5_path))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egodex-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--task-filter", default=None)
    parser.add_argument("--video-mode", choices=("symlink", "copy", "skip"), default="symlink")
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--start-id", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    items = source_items(args.egodex_root)
    if args.task_filter:
        items = [item for item in items if item[0] == args.task_filter]
    if args.max_videos is not None:
        items = items[: args.max_videos]
    if not items:
        raise RuntimeError(f"No EgoDex mp4+hdf5 pairs found under {args.egodex_root}")
    print(f"Found {len(items)} EgoDex mp4+hdf5 pairs.", flush=True)

    counters: defaultdict[str, int] = defaultdict(lambda: args.start_id)
    manifest_rows: list[dict[str, Any]] = []
    for index, (task, video_path, hdf5_path) in enumerate(items, start=1):
        item_started = time.time()
        item_id = counters[task]
        counters[task] += 1
        item_id_str = str(item_id)

        out_video = args.output_root / "video" / task / f"{item_id_str}.mp4"
        out_asset = args.output_root / "hand_asset" / task / f"{item_id_str}.npz"
        out_asset_meta = args.output_root / "hand_asset" / task / f"{item_id_str}.meta.json"
        out_raw = args.output_root / "raw_annotation" / task / f"{item_id_str}.json"

        link_or_copy(video_path.resolve(), out_video, args.video_mode)
        asset_meta = write_hand_asset(hdf5_path, out_asset, args.confidence_threshold)
        asset_meta.update(
            {
                "video": str(out_video.relative_to(args.output_root)),
                "pose_file": str(out_asset.relative_to(args.output_root)),
                "source_video": str(video_path),
            }
        )
        write_json(out_asset_meta, asset_meta)

        raw_annotation = {
            "episode_annotation": {
                "instruction": readable_task(task),
            },
            "source_dataset": "EgoDex",
            "source_video": str(video_path),
            "source_hdf5": str(hdf5_path),
            "source_task": task,
        }
        write_json(out_raw, raw_annotation)

        manifest_rows.append(
            {
                "task": task,
                "id": item_id_str,
                "video": str(out_video.relative_to(args.output_root)),
                "hand_asset": str(out_asset.relative_to(args.output_root)),
                "raw_annotation": str(out_raw.relative_to(args.output_root)),
                "source_video": str(video_path),
                "source_hdf5": str(hdf5_path),
            }
        )
        if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(items)):
            elapsed = time.time() - started
            item_elapsed = time.time() - item_started
            print(
                f"[{index}/{len(items)}] task={task} id={item_id_str} "
                f"item_sec={item_elapsed:.2f} elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    write_json(args.output_root / "meta.json", {"dataset": "EgoDex", "items": manifest_rows})
    manifest_path = args.output_root / "manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("task\tid\tvideo\thand_asset\traw_annotation\tsource_video\tsource_hdf5\n")
        for row in manifest_rows:
            f.write(
                "\t".join(
                    [
                        row["task"],
                        row["id"],
                        row["video"],
                        row["hand_asset"],
                        row["raw_annotation"],
                        row["source_video"],
                        row["source_hdf5"],
                    ]
                )
                + "\n"
            )
    print(f"Done: {args.output_root}", flush=True)
    print(f"Total seconds: {time.time() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
