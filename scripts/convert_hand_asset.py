#!/usr/bin/env python3
"""Convert an EgoANT standard hand asset into lightweight hand_recon.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_video_info(video: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if nframes <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video}")
    return nframes, fps, width, height


def fit_array(value: np.ndarray | None, nframes: int, dims: int, fill: float = np.nan) -> np.ndarray:
    out = np.full((nframes, dims), fill, dtype=float)
    if value is None:
        return out
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < dims:
        return out
    count = min(nframes, arr.shape[0])
    out[:count] = arr[:count, :dims]
    return out


def fit_bool(value: np.ndarray | None, nframes: int) -> np.ndarray:
    out = np.zeros(nframes, dtype=bool)
    if value is None:
        return out
    arr = np.asarray(value, dtype=bool).reshape(-1)
    count = min(nframes, arr.shape[0])
    out[:count] = arr[:count]
    return out


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=float)
    homogeneous = np.concatenate([points, ones], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def project(points_cam: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    xy = np.full((points_cam.shape[0], 2), np.nan, dtype=float)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-6)
    if not valid.any():
        return xy
    z = points_cam[valid, 2]
    xy[valid, 0] = intrinsics[0, 0] * (points_cam[valid, 0] / z) + intrinsics[0, 2]
    xy[valid, 1] = intrinsics[1, 1] * (points_cam[valid, 1] / z) + intrinsics[1, 2]
    return xy


def infer_camera_convention(
    camera_transform: np.ndarray,
    points_world: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
) -> str:
    if camera_transform.size == 0 or points_world.size == 0:
        return "camera_to_world"

    sample_count = min(len(points_world), len(camera_transform), 200)
    idx = np.linspace(0, sample_count - 1, min(sample_count, 40)).round().astype(int)
    points = points_world[idx]
    cams = camera_transform[idx]
    scores: dict[str, int] = {}
    for convention in ("camera_to_world", "world_to_camera"):
        count = 0
        for cam, point in zip(cams, points):
            if not np.isfinite(cam).all() or not np.isfinite(point).all():
                continue
            matrix = np.linalg.inv(cam) if convention == "camera_to_world" else cam
            xy = project(transform_points(matrix, point[None, :]), intrinsics)[0]
            if np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height:
                count += 1
        scores[convention] = count
    return max(scores, key=scores.get)


def project_world_track(
    points_world: np.ndarray,
    camera_transform: np.ndarray | None,
    intrinsics: np.ndarray | None,
    width: int,
    height: int,
    convention: str,
) -> np.ndarray:
    xy = np.full((len(points_world), 2), np.nan, dtype=float)
    if camera_transform is None or intrinsics is None:
        return xy

    if convention == "auto":
        convention = infer_camera_convention(camera_transform, points_world, intrinsics, width, height)

    count = min(len(points_world), len(camera_transform))
    for i in range(count):
        point = points_world[i]
        cam = camera_transform[i]
        if not np.isfinite(point).all() or not np.isfinite(cam).all():
            continue
        matrix = np.linalg.inv(cam) if convention == "camera_to_world" else cam
        xy[i] = project(transform_points(matrix, point[None, :]), intrinsics)[0]
    return xy


def hand_track(
    asset: dict[str, np.ndarray],
    hand: str,
    nframes: int,
    width: int,
    height: int,
    convention: str,
) -> dict[str, Any]:
    valid = fit_bool(asset.get(f"{hand}_valid"), nframes)
    transform = asset.get(f"{hand}_hand_transform")
    joints_2d = asset.get(f"{hand}_joints_2d")
    camera_transform = asset.get("camera_transform")
    intrinsics = asset.get("camera_intrinsic")

    wrist_world = np.full((nframes, 3), np.nan, dtype=float)
    if isinstance(transform, np.ndarray) and transform.ndim == 3 and transform.shape[1:] == (4, 4):
        count = min(nframes, transform.shape[0])
        wrist_world[:count] = transform[:count, :3, 3]

    if isinstance(joints_2d, np.ndarray) and joints_2d.ndim == 3 and joints_2d.shape[1] >= 1:
        wrist_xy = fit_array(joints_2d[:, 0, :2], nframes, 2)
        joints_xy = np.full((nframes, joints_2d.shape[1], 2), np.nan, dtype=float)
        count = min(nframes, joints_2d.shape[0])
        joints_xy[:count] = joints_2d[:count, :, :2]
    else:
        wrist_xy = project_world_track(wrist_world, camera_transform, intrinsics, width, height, convention)
        joints_xy = None

    valid &= np.isfinite(wrist_world).all(axis=1)
    out = {
        "valid": valid.tolist(),
        "wrist_world": wrist_world.tolist(),
        "wrist_xy": wrist_xy.tolist(),
    }
    if joints_xy is not None:
        out["joints_xy"] = joints_xy.tolist()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-asset", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--camera-convention",
        choices=("auto", "camera_to_world", "world_to_camera"),
        default="auto",
        help="How camera_transform maps coordinates. auto picks the convention with more in-frame projections.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nframes, fps, width, height = read_video_info(args.video)
    with np.load(args.hand_asset, allow_pickle=True) as data:
        asset = {key: data[key] for key in data.files}

    out = {
        "source_type": "egovid_hand_asset_npz",
        "source": str(args.hand_asset),
        "video_path": str(args.video),
        "fps": fps,
        "nframes": nframes,
        "width": width,
        "height": height,
        "camera": {
            "intrinsics": asset["camera_intrinsic"].tolist() if "camera_intrinsic" in asset else None,
            "camera_transform_shape": list(asset["camera_transform"].shape) if "camera_transform" in asset else None,
            "camera_convention": args.camera_convention,
            "width": width,
            "height": height,
        },
        "hands": {
            hand: hand_track(asset, hand, nframes, width, height, args.camera_convention)
            for hand in ("left", "right")
        },
    }
    json_dump(args.output, out)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
