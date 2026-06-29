"""
Pixel-embedding model for leaf instance segmentation.

Follows the *associative embedding* paradigm of Newell, Huang, Deng
(2016) "Associative Embedding: End-to-End Learning for Joint Detection
and Grouping" (https://arxiv.org/abs/1611.05424). For each image, the
network predicts two pixel-wise quantities:

  1. a foreground/background score (1 channel, sigmoid)
  2. an identity tag (1 channel, real-valued scalar)

The tags are trained with a *grouping loss* that pulls tags of pixels
belonging to the same leaf instance toward a common mean and pushes
the means of different instances apart. At inference time, foreground
pixels are clustered in the (1-D) tag space (mean-shift) to recover
individual leaf instances.

Architecture
------------
We re-use the ResNet-50 + FPN backbone from torchvision's
`maskrcnn_resnet50_fpn_v2` (COCO V1 weights) so this model and the
Mask R-CNN baseline compare on equal footing. The FPN's highest-
resolution feature map (P2, stride 4) is fed through a small 2-conv
shared trunk and split into two 1×1 conv heads — one for the semantic
score, one for the embedding tag — followed by bilinear upsampling to
the input resolution.

Compared to the stacked-hourglass architecture of the original Newell
paper, this swap is deliberate: the loss formulation and the grouping
behaviour are architecture-agnostic, and reusing the Mask R-CNN
backbone gives a fair paradigm comparison (detect-then-segment vs
segment-then-group) on identical pretrained weights.

Loss
----
The loss follows Newell et al. 2016 Section 3.4 (Instance Segmentation)
**byte-for-byte**:

    L_pull(per image)  = Σ_n Σ_{x,x' ∈ S_n} (h(x) - h(x'))²
    L_push(per image)  = Σ_{n≠n'} Σ_{x ∈ S_n, x' ∈ S_{n'}}
                            exp(-(h(x) - h(x'))² / 2σ²)
    L_detection        = MSE between the predicted detection heatmap
                         (sigmoid of semantic logits) and the union of
                         instance masks

    L_total = λ_pull · L_pull + λ_push · L_push + λ_det · L_detection

Where S_n is a set of K randomly-sampled pixel locations inside the
n-th object instance (paper recommends a "small set"; we use K=20 by
default). Pairwise sums are normalised by the number of pairs so the
loss magnitude stays roughly stable as the number of instances per
image varies.

Detection: paper uses MSE on the heatmap "without sigmoid" but for
numerical stability we apply sigmoid to the logits before MSE (target
is the {0,1} union of instance masks). This is mathematically a
constrained version of MSE-on-raw with the same gradient direction.

Usage
-----
    from final_project.models.pixel_embed import (
        build_pixel_embed_model, TaggingLoss
    )

    model = build_pixel_embed_model(pretrained=True)
    loss_fn = TaggingLoss(sigma=1.0, lambda_semantic=1.0)

    preds  = model(images)        # list of dicts: {'semantic', 'embedding'}
    losses = loss_fn(preds, targets)  # dict: {pull, push, semantic}
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


# ════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════
class PixelEmbeddingModel(nn.Module):
    """ResNet-50 + FPN + 2 heads (semantic + 1-D embedding tag)."""

    # ImageNet stats (same as torchvision MaskRCNN's internal normaliser)
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained: bool = True,
        trainable_backbone_layers: int = 3,
        head_channels: int = 256,
    ):
        super().__init__()
        weights = (
            MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
        )
        full_model = maskrcnn_resnet50_fpn_v2(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
        # The FPN backbone returns a dict of multi-scale features.
        # Keys: '0','1','2','3' (P2..P5) and 'pool' (P6). All 256ch.
        self.backbone = full_model.backbone

        # Small shared 2-conv trunk on P2 (highest resolution).
        self.shared = nn.Sequential(
            nn.Conv2d(256, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, head_channels),
            nn.ReLU(inplace=True),
        )
        # Two 1×1 conv heads.
        self.semantic_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(head_channels, 1, kernel_size=1)

        # Pre-compute normalisation buffers so they move with .to(device).
        self.register_buffer(
            "_mean",
            torch.tensor(self.IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(self.IMAGE_STD, dtype=torch.float32).view(3, 1, 1),
        )

    # ----------------------------------------------------------------
    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self._mean) / self._std

    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        """Run one image at a time (sizes vary in our dataset).

        Returns a list of dicts, one per input image:
            {
              'semantic':  (H, W) torch.float32 logits
              'embedding': (H, W) torch.float32 real-valued tag
            }
        """
        outputs: list[dict] = []
        for image in images:
            if image.dim() != 3 or image.shape[0] != 3:
                raise ValueError(
                    f"expected (3, H, W) image, got {tuple(image.shape)}"
                )
            H, W = image.shape[-2:]
            x = self._normalize(image).unsqueeze(0)         # (1, 3, H, W)
            features = self.backbone(x)                       # dict
            p2 = features["0"]                               # (1, 256, H/4, W/4)

            shared = self.shared(p2)
            sem = self.semantic_head(shared)                  # (1, 1, H/4, W/4)
            emb = self.embedding_head(shared)                 # (1, 1, H/4, W/4)

            sem = F.interpolate(sem, size=(H, W), mode="bilinear", align_corners=False)
            emb = F.interpolate(emb, size=(H, W), mode="bilinear", align_corners=False)

            outputs.append({
                "semantic":  sem[0, 0],   # (H, W)
                "embedding": emb[0, 0],   # (H, W)
            })
        return outputs


def build_pixel_embed_model(
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
) -> PixelEmbeddingModel:
    """Factory mirroring `final_project.models.mask_rcnn.build_maskrcnn`."""
    return PixelEmbeddingModel(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
    )


# ════════════════════════════════════════════════════════════════════
# Loss — Newell associative-embedding / tagging loss
# ════════════════════════════════════════════════════════════════════
class TaggingLoss(nn.Module):
    """Pull + push + MSE-detection loss for pixel-wise associative embedding.

    Implements Newell et al. 2016 Section 3.4 (Instance Segmentation)
    formula:

        L_pull = Σ_n Σ_{x,x' ∈ S_n} (h(x) - h(x'))²
        L_push = Σ_{n,n'} Σ_{x ∈ S_n, x' ∈ S_{n'}}
                    exp(-(h(x) - h(x'))² / 2σ²)

    where S_n is a random sample of `n_sample` pixels from the n-th
    instance mask. Pairwise sums are normalised by the number of pairs
    (per-instance for pull, per-instance-pair for push) so the loss
    magnitude does not scale with K² or the number of instances.

    Args:
        sigma: Gaussian bandwidth for the push term.
        n_sample: K — number of pixels sampled per instance for the
            pairwise pull/push terms (paper recommends "a small set").
        min_pixels: instances with fewer than this many pixels are
            skipped; tiny instances would otherwise yield noisy samples.
        lambda_pull, lambda_push, lambda_detection: scalar weights.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        n_sample: int = 20,
        min_pixels: int = 10,
        lambda_pull: float = 1.0,
        lambda_push: float = 1.0,
        lambda_detection: float = 1.0,
    ):
        super().__init__()
        self.sigma = sigma
        self.n_sample = n_sample
        self.min_pixels = min_pixels
        self.lambda_pull = lambda_pull
        self.lambda_push = lambda_push
        self.lambda_detection = lambda_detection

    def _sample_tags(
        self, tag_map: torch.Tensor, mask: torch.Tensor, k: int
    ) -> torch.Tensor | None:
        """Randomly sample k tag values from inside a single instance mask.
        Returns a (k,) tensor or None if the mask is too small."""
        ys, xs = torch.where(mask)
        n_px = ys.numel()
        if n_px < self.min_pixels:
            return None
        if n_px >= k:
            idx = torch.randperm(n_px, device=mask.device)[:k]
        else:
            idx = torch.randint(0, n_px, (k,), device=mask.device)
        return tag_map[ys[idx], xs[idx]]

    def forward(self, predictions: list[dict], targets: list[dict]) -> dict:
        """
        predictions: list of dicts with keys 'semantic' and 'embedding',
            each a (H, W) tensor sized to the original image.
        targets: list of dicts with 'masks' tensor (N, H, W) uint8/bool.
        """
        device = predictions[0]["semantic"].device
        pull_sum = torch.zeros((), device=device)
        push_sum = torch.zeros((), device=device)
        det_sum = torch.zeros((), device=device)
        n_imgs = 0
        n_imgs_with_inst = 0
        n_imgs_with_pairs = 0

        for pred, tgt in zip(predictions, targets):
            sem_logit = pred["semantic"]            # (H, W)
            tag_map = pred["embedding"]             # (H, W)

            masks = tgt["masks"]
            if not torch.is_tensor(masks):
                continue
            masks = masks.to(device=device, dtype=torch.float32)
            if masks.numel() == 0:
                continue

            # ── Detection loss: MSE on sigmoid(heatmap) vs union of masks
            # Paper uses MSE on the heatmap directly; we apply sigmoid
            # first for numerical stability (target is binary {0,1}).
            det_target = (masks.sum(dim=0) > 0).float()
            det_pred = torch.sigmoid(sem_logit)
            det_sum = det_sum + F.mse_loss(det_pred, det_target, reduction="mean")

            # ── Sample K pixels per instance
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

            # ── Pull: pairwise (h(x) - h(x'))² inside each instance
            # Normalised per pair (mean over K² pairs), then per instance.
            pull_per_img = torch.zeros((), device=device)
            for tags in sampled:
                diff = tags.unsqueeze(0) - tags.unsqueeze(1)     # (K, K)
                pull_per_img = pull_per_img + (diff ** 2).mean()
            pull_per_img = pull_per_img / n_valid
            pull_sum = pull_sum + pull_per_img
            n_imgs_with_inst += 1

            # ── Push: pairwise exp(-d² / 2σ²) across instance pairs
            # Normalised per pixel-pair, then per instance-pair.
            if n_valid >= 2:
                all_tags = torch.stack(sampled)                   # (N_inst, K)
                push_per_img = torch.zeros((), device=device)
                n_inst_pairs = 0
                two_sigma2 = 2.0 * self.sigma ** 2
                for i in range(n_valid):
                    for j in range(i + 1, n_valid):
                        d = all_tags[i].unsqueeze(0) - all_tags[j].unsqueeze(1)
                        push_per_img = push_per_img + torch.exp(
                            -(d * d) / two_sigma2
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


# ════════════════════════════════════════════════════════════════════
# CLI smoke test  —  `python -m final_project.models.pixel_embed`
# ════════════════════════════════════════════════════════════════════
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
    # Two dummy images of different sizes (mirrors our variable-sized crops)
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

    # Quick loss sanity — fabricate 3 instance masks per image
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

    # Ensure backward works
    losses["total"].backward()
    print("\nbackward pass ok.")
