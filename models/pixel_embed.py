from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


class PixelEmbeddingModel(nn.Module):
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        head_channels: int = 256,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        )
        full_model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        self.backbone = full_model.backbone
        self.shared = nn.Sequential(
            nn.Conv2d(256, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, head_channels),
            nn.ReLU(inplace=True),
        )
        self.semantic_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(head_channels, embedding_dim, kernel_size=1)
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

            shared = self.shared(p2)
            sem = self.semantic_head(shared)                  
            emb = self.embedding_head(shared)                 
            sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)
            emb = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)
            outputs.append({
                "semantic":  sem[0, 0],   
                "embedding": emb[0],      
            })
        return outputs


def build_pixel_embed_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
    embedding_dim: int = 16,
) -> PixelEmbeddingModel:
    return PixelEmbeddingModel(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
        embedding_dim=embedding_dim,
    )


class TaggingLoss(nn.Module):
    def __init__(
        self,
        sigma: float = 1.0,
        n_sample: int = 20,
        min_pixels: int = 10,
        lambda_pull: float = 1.0,
        lambda_push: float = 1.0,
        lambda_detection: float = 1.0,
        detection_loss: str = "bce",
    ):
        super().__init__()
        if detection_loss not in {"bce", "mse"}:
            raise ValueError(
                f"detection_loss must be 'bce' or 'mse', got {detection_loss!r}"
            )
        self.sigma = sigma
        self.n_sample = n_sample
        self.min_pixels = min_pixels
        self.lambda_pull = lambda_pull
        self.lambda_push = lambda_push
        self.lambda_detection = lambda_detection
        self.detection_loss = detection_loss

    def _sample_tags(
        self, tag_map: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor | None:
        ys, xs = torch.where(mask)
        n_px = ys.numel()
        if n_px < self.min_pixels:
            return None
        if n_px >= k:
            idx = torch.randperm(n_px, device=mask.device)[:k]
        else:
            idx = torch.randint(0, n_px, (k,), device=mask.device)
        
        return tag_map[:, ys[idx], xs[idx]].t()
    
    def forward(self, predictions: list[dict], targets: list[dict]) -> dict:
        device = predictions[0]["semantic"].device
        pull_sum = torch.zeros((), device=device)
        push_sum = torch.zeros((), device=device)
        det_sum = torch.zeros((), device=device)
        n_imgs = 0
        n_imgs_with_inst = 0
        n_imgs_with_pairs = 0
        for pred, tgt in zip(predictions, targets):
            sem_logit = pred["semantic"]            
            tag_map = pred["embedding"]             

            masks = tgt["masks"]
            if not torch.is_tensor(masks):
                continue
            masks = masks.to(device=device, dtype=torch.float32)
            if masks.numel() == 0:
                continue
            det_target = (masks.sum(dim=0) > 0).float()
            if self.detection_loss == "mse":
                det_sum = det_sum + F.mse_loss(
                    sem_logit, det_target, reduction="mean"
                )
            else:
                det_sum = det_sum + F.binary_cross_entropy_with_logits(
                    sem_logit, det_target, reduction="mean"
                )
            sampled: list[torch.Tensor] = []
            for k in range(masks.shape[0]):
                tags_k = self._sample_tags(
                    tag_map, masks[k] > 0.5, self.n_sample
                )
                if tags_k is not None:
                    sampled.append(tags_k)
            n_valid = len(sampled)
            if n_valid == 0:
                n_imgs += 1
                continue
            pull_per_img = torch.zeros((), device=device)
            for tags in sampled:
                diff = tags.unsqueeze(0) - tags.unsqueeze(1)     
                pull_per_img = pull_per_img + (diff ** 2).sum(-1).mean()
            pull_per_img = pull_per_img / n_valid
            pull_sum = pull_sum + pull_per_img
            n_imgs_with_inst += 1
            if n_valid >= 2:
                all_tags = torch.stack(sampled)                   
                push_per_img = torch.zeros((), device=device)
                n_inst_pairs = 0
                two_sigma2 = 2.0 * self.sigma ** 2
                for i in range(n_valid):
                    for j in range(i + 1, n_valid):
                        d = all_tags[i].unsqueeze(0) - all_tags[j].unsqueeze(1)  
                        d2 = (d * d).sum(-1)                                     
                        push_per_img = push_per_img + torch.exp(
                            -d2 / two_sigma2
                        ).mean()
                        n_inst_pairs += 1
                push_per_img = push_per_img / n_inst_pairs
                push_sum = push_sum + push_per_img
                n_imgs_with_pairs += 1
            n_imgs += 1
        loss_pull = pull_sum / max(n_imgs_with_inst, 1)
        loss_push = push_sum / max(n_imgs_with_pairs, 1)
        loss_det = det_sum / max(n_imgs, 1)

        total = (
            self.lambda_pull * loss_pull
            + self.lambda_push * loss_push
            + self.lambda_detection * loss_det
        )
        return {
            "loss_pull": loss_pull,
            "loss_push": loss_push,
            "loss_detection": loss_det,
            "total": total,
        }


if __name__ == "__main__":
    print("=== building PixelEmbeddingModel ===")
    model = build_pixel_embed_model(pretrained=True)
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"device: {device}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params  total: {total:>13,}")
    print(f"params  train: {trainable:>13,}")
    print(f"params  freeze: {total - trainable:>13,}")
    model.to(device).eval()
    images = [
        torch.rand(3, 320, 320, device=device),
        torch.rand(3, 480, 600, device=device),
    ]
    with torch.no_grad():
        preds = model(images)
    print("\n=== forward pass (eval) ===")
    for i, p in enumerate(preds):
        print(f"  image {i}: semantic {tuple(p['semantic'].shape)}  "
              f"embedding {tuple(p['embedding'].shape)}")
    print("\n=== loss sanity (train mode) ===")
    model.train()
    targets = []
    for img in images:
        H, W = img.shape[-2:]
        m = torch.zeros((3, H, W), device=device, dtype=torch.uint8)
        m[0, : H // 3, :] = 1
        m[1, H // 3 : 2 * H // 3, :] = 1
        m[2, 2 * H // 3 :, :] = 1
        targets.append({"masks": m})
    preds = model(images)
    loss_fn = TaggingLoss()
    losses = loss_fn(preds, targets)
    for k, v in losses.items():
        print(f"  {k:<15}: {v.item():.4f}")
    losses["total"].backward()
    print("\nbackward pass ok.")
