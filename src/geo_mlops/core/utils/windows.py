from __future__ import annotations

from typing import List, Tuple

import torch

def build_grid(H: int, W: int, T: int, S: int) -> Tuple[List[int], List[int]]:
    ys = list(range(0, max(1, H - T + 1), S))
    if ys[-1] != H - T:
        ys.append(max(0, H - T))
    xs = list(range(0, max(1, W - T + 1), S))
    if xs[-1] != W - T:
        xs.append(max(0, W - T))
    return ys, xs

def _to_channels(x: torch.Tensor, out_channels: int) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [C,H,W], got shape {tuple(x.shape)}")

        c = int(x.shape[0])

        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        if c == out_channels:
            return x

        if c == 1 and out_channels > 1:
            return x.repeat(out_channels, 1, 1)

        if c > out_channels:
            return x[:out_channels]

        pad = out_channels - c
        return torch.cat([x, x[-1:].repeat(pad, 1, 1)], dim=0)