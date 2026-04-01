from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from pyproj import Geod
import rasterio
import numpy as np


def _relaxed_lookup(stem: str, mapping: Dict[str, Path]) -> Optional[Path]:
    p = mapping.get(stem)
    if p is not None:
        return p
    for k, v in mapping.items():
        if stem.startswith(k) or k.startswith(stem):
            return v
    return None


def compute_gsd_from_gcps(gcps) -> float:
    geod = Geod(ellps="WGS84")
    g_tl, g_bl, g_tr = gcps[0], gcps[1], gcps[3]
    # vertical (rows)
    _, _, dist_v = geod.inv(g_tl.x, g_tl.y, g_bl.x, g_bl.y)
    pix_v = abs(g_bl.row - g_tl.row)
    mpp_y = dist_v / max(1e-6, pix_v)
    # horizontal (cols)
    _, _, dist_h = geod.inv(g_tl.x, g_tl.y, g_tr.x, g_tr.y)
    pix_h = abs(g_tr.col - g_tl.col)
    mpp_x = dist_h / max(1e-6, pix_h)
    return float((mpp_x + mpp_y) / 2.0)


def gsd_from_epsg4326(path):
    geod = Geod(ellps="WGS84")

    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326, got {src.crs}")

        # pixel size in degrees
        deg_x, deg_y = src.res

        # latitude at image center (important for accuracy)
        center_lat = (src.bounds.top + src.bounds.bottom) / 2

        # meters per pixel (longitude varies with latitude)
        _, _, meters_x = geod.inv(
            0, center_lat,
            deg_x, center_lat
        )

        # latitude spacing
        _, _, meters_y = geod.inv(
            0, center_lat,
            0, center_lat + deg_y
        )

        return float((abs(meters_x) + abs(meters_y)) / 2)


def _positions(limit: int, tile: int, stride: int):
    # all regular positions
    last_start = limit - tile
    if last_start < 0:
        return [0]  # tile bigger than image => single tile anchored at 0
    xs = list(range(0, last_start + 1, stride))
    if xs[-1] != last_start:
        xs.append(last_start)
    return xs


def gen_tiles_cover(
    h: int, w: int, tile_h: int, tile_w: int, stride_h: int, stride_w: int
) -> Iterator[Tuple[int, int, int, int, int, int]]:
    ys = _positions(h, tile_h, stride_h)
    xs = _positions(w, tile_w, stride_w)

    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            yield x, y, x + tile_w, y + tile_h, row, col
