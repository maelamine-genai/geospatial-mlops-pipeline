from __future__ import annotations

from typing import List, Tuple

def build_grid(H: int, W: int, T: int, S: int) -> Tuple[List[int], List[int]]:
    ys = list(range(0, max(1, H - T + 1), S))
    if ys[-1] != H - T:
        ys.append(max(0, H - T))
    xs = list(range(0, max(1, W - T + 1), S))
    if xs[-1] != W - T:
        xs.append(max(0, W - T))
    return ys, xs