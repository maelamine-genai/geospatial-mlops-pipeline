from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm


@dataclass(frozen=True)
class PanContextConfig:
    # If set, context size = scale * (H,W)
    pan_context_scale: Optional[float] = 0.25
    # Else, scale is chosen so max(H,W) <= max_side
    pan_context_max_side: Optional[int] = 2048
    # Used for normalization to uint8 (typical for reflectance-like PAN)
    pan_context_reflectance_max: int = 10_000
    # If False, skip when output already exists
    create_pan_context_image: bool = True

    # Rasterization knobs
    resampling: Resampling = Resampling.bilinear
    all_bands: bool = False  # for PAN we default to band 1 only


def _decide_out_hw(H: int, W: int, cfg: PanContextConfig) -> Tuple[int, int, float]:
    """
    Decide output (h, w) and scale_factor.
    """
    if H <= 0 or W <= 0:
        raise ValueError(f"Invalid raster shape H={H}, W={W}")

    scale = cfg.pan_context_scale
    if scale is None:
        max_side = cfg.pan_context_max_side
        if max_side is None or max_side <= 0:
            scale = 0.25
        else:
            scale = min(1.0, float(max_side) / float(max(H, W)))

    h = max(1, int(round(H * scale)))
    w = max(1, int(round(W * scale)))
    return h, w, float(scale)


def _normalize_to_uint8(x: np.ndarray, *, ref_max: float) -> np.ndarray:
    """
    Normalize an array to uint8 using ref_max.
    Expects x as float32/float64 or integer.
    """
    x = x.astype(np.float32, copy=False)
    x = np.clip(x, 0.0, ref_max) / max(1e-6, ref_max)
    x = (x * 255.0).round().astype(np.uint8)
    return x


def save_context_tif_from_pan(
    pan_path: Path,
    out_path: Path,
    cfg: PanContextConfig,
) -> Path:
    """
    Read a (typically single-band) PAN GeoTIFF, downsample to uint8 context image,
    and write a GeoTIFF with updated transform/shape.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not cfg.create_pan_context_image:
        return out_path

    if not pan_path.exists():
        raise FileNotFoundError(f"PAN image missing: {pan_path}")

    with rasterio.open(pan_path) as src:
        H, W = src.height, src.width
        h, w, _scale = _decide_out_hw(H, W, cfg)

        # Read with resampling directly (fast and preserves georef)
        # PAN: use first band by default
        if cfg.all_bands:
            data = src.read(
                out_shape=(src.count, h, w),
                resampling=cfg.resampling,
                masked=True,
            )
            # Convert to single context band by averaging bands
            # (you can change this if you ever use multi-band context)
            data_f = np.mean(data.filled(0).astype(np.float32), axis=0)
        else:
            data = src.read(
                1,
                out_shape=(h, w),
                resampling=cfg.resampling,
                masked=True,
            )
            data_f = data.filled(0).astype(np.float32)

        # Normalize -> uint8
        ref_max = float(cfg.pan_context_reflectance_max)
        ctx_u8 = _normalize_to_uint8(data_f, ref_max=ref_max)

        # Update transform for the new resolution
        # NOTE: rasterio provides a helper pattern for scaling transforms:
        # new_transform = old_transform * old_transform.scale(old_w/new_w, old_h/new_h)
        new_transform = src.transform * src.transform.scale(
            (W / float(w)),
            (H / float(h)),
        )

        meta = src.meta.copy()
        meta.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            height=h,
            width=w,
            transform=new_transform,
            nodata=0,
            compress="deflate",
        )

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(ctx_u8, 1)

    return out_path


def main():
    ap = argparse.ArgumentParser("Generate PAN context images into sibling 'Context' dirs")

    ap.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory to search for PAN dirs (recursively).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for PAN chip files (inside PAN dirs).",
    )
    ap.add_argument(
        "--context-suffix",
        type=str,
        default="",
        help="Suffix to append before extension for context images.",
    )
    ap.add_argument(
        "--pan-context-scale",
        type=float,
        default=0.25,
        help="Optional fixed downsample scale (e.g., 0.25). If not set, max-side is used.",
    )
    ap.add_argument(
        "--pan-context-max-side",
        type=int,
        default=2048,
        help="Max side (px) for context images if scale is not set. Set <=0 to fall back to scale=0.25.",
    )
    ap.add_argument(
        "--pan-context-reflectance-max",
        type=int,
        default=10_000,
        help="Max reflectance used for normalization to uint8.",
    )
    ap.add_argument(
        "--no-create-if-exists",
        action="store_true",
        help="Skip writing context image if output already exists.",
    )
    ap.add_argument(
        "--pan-dirname",
        type=str,
        default="PAN",
        help="Directory name that contains PAN imagery.",
    )
    ap.add_argument(
        "--context-dirname",
        type=str,
        default="Context",
        help="Sibling output directory name to store context imagery.",
    )

    args = ap.parse_args()
    root_dir: Path = args.root_dir.resolve()

    cfg = PanContextConfig(
        pan_context_scale=args.pan_context_scale,
        pan_context_max_side=args.pan_context_max_side,
        pan_context_reflectance_max=args.pan_context_reflectance_max,
        create_pan_context_image=not args.no_create_if_exists,
    )

    pan_dirs = [p for p in root_dir.rglob(args.pan_dirname) if p.is_dir()]
    if not pan_dirs:
        print(f"No '{args.pan_dirname}' directories found under {root_dir}")
        return

    print(f"Found {len(pan_dirs)} '{args.pan_dirname}' directory(ies) under {root_dir}")

    num_created = 0
    num_skipped_exists = 0
    num_errors = 0

    for pan_dir in tqdm(pan_dirs, desc="PAN dirs"):
        context_dir = pan_dir.parent / args.context_dirname
        context_dir.mkdir(parents=True, exist_ok=True)

        chip_paths = sorted(pan_dir.rglob(args.pattern))
        if not chip_paths:
            continue

        for chip_path in tqdm(chip_paths, leave=False, desc=f"{pan_dir.relative_to(root_dir)}"):
            try:
                out_name = chip_path.stem + args.context_suffix + ".tif"
                out_path = context_dir / out_name

                if out_path.exists() and args.no_create_if_exists:
                    num_skipped_exists += 1
                    continue

                save_context_tif_from_pan(
                    pan_path=chip_path,
                    out_path=out_path,
                    cfg=cfg,
                )
                num_created += 1
            except Exception as e:  # noqa: BLE001
                num_errors += 1
                print(f"[ERROR] {chip_path}: {e}")

    print(f"Done. Created={num_created}, skipped_existing={num_skipped_exists}, errors={num_errors}")


if __name__ == "__main__":
    main()