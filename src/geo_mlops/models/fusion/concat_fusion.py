from typing import Optional
import torch
import torch.nn as nn

class ConcatFusionHead(nn.Module):
    """
    Concatenate (tile, ctx?, coords?) -> (optional 1x1 proj) -> inner(x).

    Contract:
      - forward(tile, ctx=None) returns logits.
      - inner MUST be single-input: inner.forward(x)
    """

    def __init__(
        self,
        *,
        inner_model: nn.Module,
        tile_channels: int = 1,
        ctx_channels: int = 1,
        add_coords: bool = True,
        expect_ctx: bool = True,
        inner_in_channels: Optional[int] = None,
        use_1x1_proj: bool = False,
    ):
        super().__init__()
        self.inner = inner_model
        self.tile_channels = int(tile_channels)
        self.ctx_channels = int(ctx_channels)
        self.add_coords = bool(add_coords)
        self.expect_ctx = bool(expect_ctx)

        # Compute fused input channels (constant)
        self.fused_in_channels = self.tile_channels \
            + (self.ctx_channels if self.expect_ctx else 0) \
            + (2 if self.add_coords else 0)

        # Optional projection to match inner expected channels
        # If not provided, default to "no projection" if channels match
        if inner_in_channels is None:
            inner_in_channels = self.fused_in_channels
        self.inner_in_channels = int(inner_in_channels)

        self.use_1x1_proj = bool(use_1x1_proj)
        if self.use_1x1_proj:
            self.proj = nn.Conv2d(self.fused_in_channels, self.inner_in_channels, kernel_size=1)
        else:
            if self.fused_in_channels != self.inner_in_channels:
                raise ValueError(
                    f"Channels mismatch: fused_in_channels={self.fused_in_channels} "
                    f"!= inner_in_channels={self.inner_in_channels}. "
                    f"Either set use_1x1_proj=True or make them equal."
                )
            self.proj = nn.Identity()

    def forward(self, tile: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # tile: (B, Ct, H, W)
        if tile.ndim != 4:
            raise ValueError(f"tile must be (B,C,H,W); got {tuple(tile.shape)}")
        b, ct, h, w = tile.shape
        if ct != self.tile_channels:
            raise ValueError(f"Expected tile_channels={self.tile_channels}, got {ct}")

        parts = [tile]

        if self.expect_ctx:
            if ctx is None:
                # Keep channels constant: add zeros context
                ctx = torch.zeros((b, self.ctx_channels, h, w), device=tile.device, dtype=tile.dtype)
            else:
                if ctx.ndim != 4:
                    raise ValueError(f"ctx must be (B,C,H,W); got {tuple(ctx.shape)}")
                if ctx.shape[0] != b:
                    raise ValueError(f"ctx batch {ctx.shape[0]} != tile batch {b}")
                if ctx.shape[1] != self.ctx_channels:
                    raise ValueError(f"Expected ctx_channels={self.ctx_channels}, got {ctx.shape[1]}")
                if ctx.shape[-2:] != (h, w):
                    # If you pass full context resized already in dataset, this is a no-op.
                    ctx = torch.nn.functional.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)
            parts.append(ctx)

        if self.add_coords:
            parts.append(self._make_coords(b, h, w, tile.device, tile.dtype))

        x = torch.cat(parts, dim=1)  # (B, fused_in_channels, H, W)
        x = self.proj(x)             # (B, inner_in_channels, H, W) or Identity

        return self.inner(x)

    @staticmethod
    def _make_coords(b: int, h: int, w: int, device, dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)  # (B,2,H,W)
        return coords