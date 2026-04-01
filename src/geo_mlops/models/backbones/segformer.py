from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SegFormerBackbone(nn.Module):
    """
    SegFormer semantic segmentation model that accepts a SINGLE tensor input:

        forward(x) where x is (B, C_in, H, W) in [0,1] (recommended).

    This module handles:
      - optional input projection from C_in -> 3 channels
      - calling HF SegFormer
      - upsampling logits back to (H,W)
    """

    def __init__(
        self,
        *,
        base_name: str,
        num_classes: int,
        image_size: int,
        in_channels: int,
        proj_to_rgb: bool = True,
    ):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        if base_name == "segformer_b0":
            name = "nvidia/segformer-b0-finetuned-ade-512-512"
        elif base_name == "segformer_b2":
            name = "nvidia/segformer-b2-finetuned-ade-512-512"
        else:
            raise ValueError(f"Unknown segformer arch: {base_name}")

        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )

        # Inform config (mostly for sanity; HF uses actual tensor shapes at runtime)
        self.segformer.config.num_labels = num_classes
        self.segformer.config.image_size = image_size

        self.in_channels = int(in_channels)
        self.proj_to_rgb = bool(proj_to_rgb)

        if self.proj_to_rgb:
            # Project arbitrary fused channels -> 3
            self.input_proj = nn.Sequential(
                nn.Conv2d(self.in_channels, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=True),
            )
            for m in self.input_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        else:
            # If you set proj_to_rgb=False, you MUST ensure x is already 3 channels.
            self.input_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, H, W)
        returns logits: (B, num_classes, H, W)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected x to be 4D (B,C,H,W), got {tuple(x.shape)}")

        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C}")

        # Clamp to valid range (helpful if you add noise aug)
        x = x.clamp(0, 1)

        x = self.input_proj(x)  # (B,3,H,W) or Identity
        out = self.segformer(pixel_values=x)
        logits = out.logits  # (B,num_classes,h',w')

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits