"""
Decoder-style pixel-embedding model for leaf instance segmentation.

This is an ablation of :mod:`final_project.models.pixel_embed`.
The original model predicts foreground/tag heatmaps at the P2 FPN
resolution (stride 4) and then upsamples the *outputs* to the input
image size.

This variant upsamples the *feature map* before prediction:

    P2: 256 × H/4 × W/4
      → upsample to H/2 × W/2
      → 3×3 conv
      → upsample to H × W
      → 3×3 conv
      → foreground head + embedding head at H × W

The aim is to test whether giving the heads higher-resolution features
improves foreground boundaries and small/overlapping leaf separation.
It is more memory-expensive than the default P2-head model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from final_project.models.pixel_embed import TaggingLoss


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    """Pick a GroupNorm group count that divides ``num_channels``."""
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class PixelEmbeddingDecoderModel(nn.Module):
    """ResNet-50 FPN + lightweight high-resolution decoder + two heads."""

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        head_channels: int = 256,
        decoder_channels: int = 64,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_channels = decoder_channels

        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        )
        full_model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.backbone = full_model.backbone

        # First adapt P2 at stride 4, then progressively upsample features.
        self.p2_adapter = nn.Sequential(
            nn.Conv2d(256, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, head_channels), head_channels),
            nn.ReLU(inplace=True),
        )
        self.up_h2 = nn.Sequential(
            nn.Conv2d(head_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, decoder_channels), decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.up_h = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, decoder_channels), decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.semantic_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(decoder_channels, embedding_dim, kernel_size=1)

        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGE_STD, dtype=torch.float32).view(3, 1, 1),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self._mean) / self._std

    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        outputs: list[dict] = []
        for image in images:
            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"expected (3, H, W) image, got {tuple(image.shape)}"
                )

            H, W = image.shape[-2:]
            x = self._normalize(image).unsqueeze(0)
            features = self.backbone(x)
            p2 = features["0"]  # (1, 256, H/4, W/4)

            feat = self.p2_adapter(p2)
            feat = F.interpolate(
                feat,
                size=((H + 1) // 2, (W + 1) // 2),
                mode="bilinear",
                align_corners=False,
            )
            feat = self.up_h2(feat)
            feat = F.interpolate(
                feat,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            feat = self.up_h(feat)

            sem = self.semantic_head(feat)
            emb = self.embedding_head(feat)

            outputs.append({
                "semantic": sem[0, 0],  # (H, W)
                "embedding": emb[0],    # (D, H, W)
            })
        return outputs


def build_pixel_embed_decoder_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
    decoder_channels: int = 64,
    embedding_dim: int = 16,
) -> PixelEmbeddingDecoderModel:
    return PixelEmbeddingDecoderModel(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
        decoder_channels=decoder_channels,
        embedding_dim=embedding_dim,
    )


class PixelEmbeddingH2DecoderModel(nn.Module):
    """ResNet-50 FPN + H/2 decoder + two heads.

    This is the memory-friendlier decoder ablation:

        P2: 256 × H/4 × W/4
          → 3×3 conv
          → upsample to H/2 × W/2
          → 3×3 conv
          → heads at H/2 × W/2
          → bilinear upsample outputs to H × W

    Compared with :class:`PixelEmbeddingDecoderModel`, this avoids
    keeping a full-resolution feature map before the prediction heads.
    """

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        head_channels: int = 256,
        decoder_channels: int = 64,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_channels = decoder_channels

        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        )
        full_model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.backbone = full_model.backbone

        self.p2_adapter = nn.Sequential(
            nn.Conv2d(256, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, head_channels), head_channels),
            nn.ReLU(inplace=True),
        )
        self.up_h2 = nn.Sequential(
            nn.Conv2d(head_channels, decoder_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, decoder_channels), decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.semantic_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(decoder_channels, embedding_dim, kernel_size=1)

        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGE_STD, dtype=torch.float32).view(3, 1, 1),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self._mean) / self._std

    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        outputs: list[dict] = []
        for image in images:
            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"expected (3, H, W) image, got {tuple(image.shape)}"
                )

            H, W = image.shape[-2:]
            x = self._normalize(image).unsqueeze(0)
            features = self.backbone(x)
            p2 = features["0"]  # (1, 256, H/4, W/4)

            feat = self.p2_adapter(p2)
            feat = F.interpolate(
                feat,
                size=((H + 1) // 2, (W + 1) // 2),
                mode="bilinear",
                align_corners=False,
            )
            feat = self.up_h2(feat)

            sem = self.semantic_head(feat)
            emb = self.embedding_head(feat)

            sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)
            emb = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)

            outputs.append({
                "semantic": sem[0, 0],  # (H, W)
                "embedding": emb[0],    # (D, H, W)
            })
        return outputs


def build_pixel_embed_h2_decoder_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
    decoder_channels: int = 64,
    embedding_dim: int = 16,
) -> PixelEmbeddingH2DecoderModel:
    return PixelEmbeddingH2DecoderModel(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
        decoder_channels=decoder_channels,
        embedding_dim=embedding_dim,
    )


class PixelEmbeddingFPNH2DecoderModel(nn.Module):
    """ResNet-50 FPN + multi-scale FPN-style H/2 decoder + two heads.

    This variant uses more of the FPN backbone than the P2-only models:

        P2: H/4  ───────────────→ 1×1 conv → decoder_channels ┐
        P3: H/8  → upsample H/4 → 1×1 conv → decoder_channels ┤
        P4: H/16 → upsample H/4 → 1×1 conv → decoder_channels ┤
        P5: H/32 → upsample H/4 → 1×1 conv → decoder_channels ┘
                                                        ↓
                                                   sum fusion
                                                        ↓
                                             3×3 conv refinement
                                                        ↓
                                             upsample feature to H/2
                                                        ↓
                                             3×3 conv refinement
                                                        ↓
                                      heads at H/2, outputs upsampled to H

    Compared with ``PixelEmbeddingH2DecoderModel``, this keeps the same
    H/2 prediction resolution but gives the heads both local detail from
    P2 and larger-context information from P3–P5.
    """

    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        head_channels: int = 256,
        decoder_channels: int = 64,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.decoder_channels = decoder_channels

        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        )
        full_model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.backbone = full_model.backbone

        # Torchvision's FPN backbone returns keys "0", "1", "2", "3"
        # corresponding to P2, P3, P4, P5. All are 256 channels.
        self.lateral_convs = nn.ModuleDict({
            key: nn.Sequential(
                nn.Conv2d(256, decoder_channels, kernel_size=1),
                _make_group_norm(decoder_channels),
                nn.ReLU(inplace=True),
            )
            for key in ("0", "1", "2", "3")
        })
        self.fuse_refine = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            _make_group_norm(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            _make_group_norm(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.up_h2 = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            _make_group_norm(decoder_channels),
            nn.ReLU(inplace=True),
        )

        self.semantic_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(decoder_channels, embedding_dim, kernel_size=1)

        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGE_STD, dtype=torch.float32).view(3, 1, 1),
        )

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self._mean) / self._std

    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        outputs: list[dict] = []
        for image in images:
            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"expected (3, H, W) image, got {tuple(image.shape)}"
                )

            H, W = image.shape[-2:]
            x = self._normalize(image).unsqueeze(0)
            features = self.backbone(x)

            p2_size = features["0"].shape[-2:]
            fused = None
            for key in ("0", "1", "2", "3"):
                feat = self.lateral_convs[key](features[key])
                if feat.shape[-2:] != p2_size:
                    feat = F.interpolate(
                        feat,
                        size=p2_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                fused = feat if fused is None else fused + feat

            feat = self.fuse_refine(fused)
            feat = F.interpolate(
                feat,
                size=((H + 1) // 2, (W + 1) // 2),
                mode="bilinear",
                align_corners=False,
            )
            feat = self.up_h2(feat)

            sem = self.semantic_head(feat)
            emb = self.embedding_head(feat)

            sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)
            emb = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)

            outputs.append({
                "semantic": sem[0, 0],  # (H, W)
                "embedding": emb[0],    # (D, H, W)
            })
        return outputs


def build_pixel_embed_fpn_h2_decoder_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
    decoder_channels: int = 64,
    embedding_dim: int = 16,
) -> PixelEmbeddingFPNH2DecoderModel:
    return PixelEmbeddingFPNH2DecoderModel(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
        decoder_channels=decoder_channels,
        embedding_dim=embedding_dim,
    )
