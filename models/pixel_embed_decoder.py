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
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class PixelEmbeddingDecoderModel(nn.Module):
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
            p2 = features["0"]  

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
                "semantic": sem[0, 0],  
                "embedding": emb[0],    
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
            p2 = features["0"]  

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
                "semantic": sem[0, 0],  
                "embedding": emb[0],    
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
                "semantic": sem[0, 0],  
                "embedding": emb[0],    
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
