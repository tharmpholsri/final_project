"""
Copy-paste augmentation transform for leaf instance segmentation.

Compatible with the `torchvision.transforms.v2`-style pipeline used in
`final_project/data/transforms.py`. Plugs in between the geometric and
photometric transforms so any pasted leaves get the same photometric
treatment as the underlying scene (but only one geometric flip).

Mechanism
---------
On each call we:

  1. Optionally trigger (probability `p_apply`)
  2. Build a *valid paste area mask* for this image:
       - if a pot-mask loader is provided → use the pot cell (Option B in
         COPY_PASTE_PLAN.md §5)
       - else → dilate the union of the existing leaf masks
                (fallback used for CVPPP images that have no pot mask)
  3. Sample `n_paste` leaves from the leaf bank (`leaf_bank/`)
  4. For each leaf:
       a. Random flip + rotate (any angle) + scale (e.g. 0.7×–1.3×)
       b. Try up to N positions until the placed mask overlaps the valid
          area by at least `paste_overlap_min`
       c. Composite the leaf onto the image, add a new mask + box + label
          to the target, and subtract the new mask from previously placed
          masks (the pasted leaf is in front)

Constructor
-----------
    cp = CopyPaste(
        leaf_bank_dir="leaf_bank",
        pot_mask_loader=make_pot_loader(coco_path, pots_dir),  # optional
        n_paste_range=(1, 4),
        scale_range=(0.7, 1.3),
        p_apply=0.5,
    )

    # plug into the augmentation pipeline
    train_transforms = get_train_transforms(copy_paste=cp)

`make_pot_loader` is a small factory that returns a callable
`(image_id, image_shape) -> Optional[np.ndarray]`. For CVPPP images it
returns None so the transform falls back to the dilated-existing-leaves
constraint.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from final_project.eval.metrics import stage_from_filename


# ════════════════════════════════════════════════════════════════════
# Factories — closures that bind a COCO instance for image_id lookups
# ════════════════════════════════════════════════════════════════════
def make_filename_lookup(
    coco_path: str | Path,
) -> Callable[[int], Optional[str]]:
    """Return a function that maps `image_id` to the COCO `file_name`."""
    from pycocotools.coco import COCO
    coco = COCO(str(coco_path))

    def lookup(image_id: int) -> Optional[str]:
        info = coco.imgs.get(image_id)
        if info is None:
            return None
        return info["file_name"]

    return lookup


def make_pot_loader(
    coco_path: str | Path,
    pots_dir: str | Path,
) -> Callable[[int, tuple], Optional[np.ndarray]]:
    """Return a function that maps `image_id` to the pot-cell mask for
    that image, or `None` if no pot mask exists (e.g. CVPPP).

    Loaded image masks are NOT cached because the per-image arrays are
    small and we don't want stale state across epochs."""
    from pycocotools.coco import COCO
    coco = COCO(str(coco_path))
    pots_dir = Path(pots_dir)

    def loader(image_id: int, image_shape: tuple) -> Optional[np.ndarray]:
        info = coco.imgs.get(image_id)
        if info is None:
            return None
        candidate = pots_dir / Path(info["file_name"]).name
        if not candidate.exists():
            return None
        arr = np.array(Image.open(candidate))
        if arr.shape[:2] != image_shape[:2]:
            return None  # size mismatch — refuse rather than misalign
        return (arr > 0).astype(np.uint8)

    return loader


# ════════════════════════════════════════════════════════════════════
# In-memory leaf bank
# ════════════════════════════════════════════════════════════════════
class LeafBank:
    """Load all leaf cutouts + masks into RAM for fast random sampling.

    Stamps are indexed both globally and per source_stage so callers can
    restrict sampling to a single stage (e.g. only `late` leaves) for
    stage-matched copy-paste.
    """

    def __init__(self, bank_dir: str | Path):
        bank_dir = Path(bank_dir)
        meta_path = bank_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"leaf bank not found at {bank_dir} "
                f"(did you run build_leaf_bank.py?)"
            )

        self.dir = bank_dir
        self.metadata: list[dict] = json.loads(meta_path.read_text())
        self.images: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.by_stage: dict[str, list[int]] = {}

        for i, m in enumerate(self.metadata):
            img = np.array(
                Image.open(bank_dir / "leaves" / f"{m['idx']}.png").convert("RGB")
            )
            msk = np.array(
                Image.open(bank_dir / "masks" / f"{m['idx']}.png")
            )
            self.images.append(img)
            self.masks.append((msk > 0).astype(np.uint8))
            stage = m.get("source_stage", "unknown")
            self.by_stage.setdefault(stage, []).append(i)

        if not self.images:
            raise ValueError(f"leaf bank {bank_dir} is empty")

    def __len__(self) -> int:
        return len(self.images)

    def stages(self) -> list[str]:
        return sorted(self.by_stage.keys())

    def sample(
        self,
        rng: random.Random,
        stage: Optional[str] = None,
    ) -> Optional[tuple[np.ndarray, np.ndarray, dict]]:
        """Sample one stamp uniformly. If `stage` is given, restrict to
        that bucket. Returns None if the requested stage has no leaves."""
        if stage is None:
            i = rng.randrange(len(self))
        else:
            bucket = self.by_stage.get(stage)
            if not bucket:
                return None
            i = rng.choice(bucket)
        return self.images[i], self.masks[i], self.metadata[i]


# ════════════════════════════════════════════════════════════════════
# Geometric augmentation of a single leaf
# ════════════════════════════════════════════════════════════════════
def transform_leaf(
    img: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    scale_range: tuple[float, float] = (0.7, 1.3),
    rotation_range: tuple[float, float] = (-180.0, 180.0),
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Random flip + rotate + scale on a leaf cutout.

    Returns possibly-larger arrays because rotation with `expand=True`
    grows the canvas. Mask stays aligned with image."""
    pil_img = Image.fromarray(img)
    pil_mask = Image.fromarray((mask * 255).astype(np.uint8))

    if rng.random() < p_hflip:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        pil_mask = pil_mask.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < p_vflip:
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)
        pil_mask = pil_mask.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(*rotation_range)
    pil_img = pil_img.rotate(angle, resample=Image.BILINEAR, expand=True)
    pil_mask = pil_mask.rotate(angle, resample=Image.NEAREST, expand=True)

    scale = rng.uniform(*scale_range)
    if scale != 1.0:
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        pil_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)

    new_img = np.array(pil_img)
    new_mask = (np.array(pil_mask) > 127).astype(np.uint8)

    # Tighten to mask bbox so we don't pad with extra zeros
    ys, xs = np.where(new_mask)
    if len(ys) == 0:
        return new_img, new_mask
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return new_img[y0:y1, x0:x1], new_mask[y0:y1, x0:x1]


# ════════════════════════════════════════════════════════════════════
# CopyPaste transform
# ════════════════════════════════════════════════════════════════════
class CopyPaste:
    """v2-style transform that pastes random leaves from the bank.

    Designed to slot into the pipeline returned by
    `final_project.data.transforms.get_train_transforms(copy_paste=...)`.
    """

    def __init__(
        self,
        leaf_bank_dir: str | Path = "leaf_bank",
        pot_mask_loader: Optional[Callable[[int, tuple], Optional[np.ndarray]]] = None,
        filename_from_image_id: Optional[Callable[[int], Optional[str]]] = None,
        stage_match: str = "strict",
        n_paste_range: tuple[int, int] = (1, 4),
        scale_range: tuple[float, float] = (0.7, 1.3),
        rotation_range: tuple[float, float] = (-180.0, 180.0),
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        paste_overlap_min: float = 0.95,
        max_leaf_to_image_ratio: float = 0.30,
        max_position_attempts: int = 20,
        fallback_dilation_px: int = 30,
        p_apply: float = 0.5,
        seed: Optional[int] = None,
    ):
        """
        stage_match:
          - "strict": map target filename → stage via
            `stage_from_filename`, then sample only from bank leaves of
            that stage. If the bank has no leaves for that stage (e.g.
            "early"), no paste happens on that image.
          - "off": ignore stage; sample uniformly from the whole bank
            (original behaviour). Use this for non-PACE training sources
            whose filenames don't carry a stage tag (e.g. CVPPP).
        """
        if stage_match not in ("strict", "off"):
            raise ValueError(f"stage_match must be 'strict' or 'off', got {stage_match!r}")
        self.bank = LeafBank(leaf_bank_dir)
        self.pot_loader = pot_mask_loader
        self.filename_from_image_id = filename_from_image_id
        self.stage_match = stage_match
        self.n_paste_range = n_paste_range
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.paste_overlap_min = paste_overlap_min
        self.max_leaf_to_image_ratio = max_leaf_to_image_ratio
        self.max_position_attempts = max_position_attempts
        self.fallback_dilation_px = fallback_dilation_px
        self.p_apply = p_apply
        self.rng = random.Random(seed) if seed is not None else random.Random()

    # ── helpers ─────────────────────────────────────────────────────
    def _valid_paste_area(
        self, image_shape: tuple, target: dict, image_id: int
    ) -> np.ndarray:
        """Return a binary mask (H, W) of where leaves are allowed to land.

        Preference order:
          1. pot mask (if loader returns one)
          2. dilated union of existing leaf masks
          3. whole image (last resort)
        """
        H, W = image_shape[:2]

        if self.pot_loader is not None:
            pot = self.pot_loader(image_id, (H, W))
            if pot is not None and pot.sum() > 0:
                return pot

        existing = target.get("masks")
        if isinstance(existing, torch.Tensor) and existing.numel() > 0:
            union = (existing.sum(dim=0).cpu().numpy() > 0).astype(np.uint8)
            if union.sum() > 0:
                return _dilate(union, self.fallback_dilation_px)

        return np.ones((H, W), dtype=np.uint8)

    def _try_paste_position(
        self, paste_area: np.ndarray, leaf_mask: np.ndarray
    ) -> Optional[tuple[int, int]]:
        """Find a top-left (y, x) where the leaf mask overlaps `paste_area`
        by at least `paste_overlap_min` of the leaf's own pixels.
        Returns None if no valid position is found in N tries."""
        H, W = paste_area.shape
        lh, lw = leaf_mask.shape
        if lh >= H or lw >= W:
            return None
        leaf_px = int(leaf_mask.sum())
        if leaf_px == 0:
            return None

        for _ in range(self.max_position_attempts):
            y = self.rng.randrange(0, H - lh + 1)
            x = self.rng.randrange(0, W - lw + 1)
            inside = paste_area[y:y + lh, x:x + lw] & leaf_mask
            if inside.sum() / leaf_px >= self.paste_overlap_min:
                return (y, x)
        return None

    # ── main ────────────────────────────────────────────────────────
    def __call__(self, image, target: dict):
        if self.rng.random() >= self.p_apply:
            return image, target

        # Convert image to HWC uint8 numpy for paste operations.
        # `image` is a tv_tensors.Image (CHW float [0,1]).
        if isinstance(image, torch.Tensor):
            img_chw_float = image.detach()
            arr = (img_chw_float.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        else:
            arr = np.asarray(image)

        H, W = arr.shape[:2]

        # Image id (for pot loader and stage lookup)
        img_id_t = target.get("image_id")
        image_id = int(img_id_t.item()) if isinstance(img_id_t, torch.Tensor) and img_id_t.numel() else -1

        # Resolve stage from filename (if we can). Used only when
        # stage_match == "strict".
        target_stage: Optional[str] = None
        if self.stage_match == "strict":
            if self.filename_from_image_id is not None and image_id >= 0:
                fname = self.filename_from_image_id(image_id)
                if fname:
                    target_stage = stage_from_filename(fname)
            # If we could not resolve a stage (no filename map, unknown
            # pattern, or filename not from PACE) → skip pasting entirely.
            if target_stage is None:
                return image, target
            # If the bank has no leaves for this stage → also skip
            if not self.bank.by_stage.get(target_stage):
                return image, target

        paste_area = self._valid_paste_area(arr.shape, target, image_id)

        # Existing instance masks as a numpy (N, H, W) uint8 stack
        existing_masks = target.get("masks")
        if isinstance(existing_masks, torch.Tensor) and existing_masks.numel() > 0:
            masks_np = existing_masks.detach().cpu().numpy().astype(np.uint8)
        else:
            masks_np = np.zeros((0, H, W), dtype=np.uint8)

        n_paste = self.rng.randint(*self.n_paste_range)
        new_masks: list[np.ndarray] = []
        new_boxes: list[list[float]] = []

        for _ in range(n_paste):
            sampled = self.bank.sample(self.rng, stage=target_stage)
            if sampled is None:
                continue
            leaf_img, leaf_mask, _meta = sampled
            leaf_img, leaf_mask = transform_leaf(
                leaf_img, leaf_mask, self.rng,
                scale_range=self.scale_range,
                rotation_range=self.rotation_range,
                p_hflip=self.p_hflip,
                p_vflip=self.p_vflip,
            )
            if leaf_mask.shape[0] == 0 or leaf_mask.shape[1] == 0:
                continue

            # Cap leaf size relative to the target image so canopy stamps
            # don't dominate small early-stage scenes.
            target_max = int(min(H, W) * self.max_leaf_to_image_ratio)
            leaf_long = max(leaf_mask.shape[:2])
            if leaf_long > target_max and target_max > 0:
                shrink = target_max / leaf_long
                new_h = max(1, int(leaf_mask.shape[0] * shrink))
                new_w = max(1, int(leaf_mask.shape[1] * shrink))
                pil_l = Image.fromarray(leaf_img).resize((new_w, new_h), Image.BILINEAR)
                pil_m = Image.fromarray((leaf_mask * 255).astype(np.uint8)).resize(
                    (new_w, new_h), Image.NEAREST
                )
                leaf_img = np.array(pil_l)
                leaf_mask = (np.array(pil_m) > 127).astype(np.uint8)
                if leaf_mask.sum() == 0:
                    continue

            pos = self._try_paste_position(paste_area, leaf_mask)
            if pos is None:
                continue
            y, x = pos
            lh, lw = leaf_mask.shape

            # Paste image (alpha blend using mask as alpha)
            patch_arr = arr[y:y + lh, x:x + lw]
            alpha = leaf_mask[..., None].astype(np.float32)
            arr[y:y + lh, x:x + lw] = (
                patch_arr * (1 - alpha) + leaf_img * alpha
            ).astype(np.uint8)

            # Build full-size mask for the newly pasted leaf
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[y:y + lh, x:x + lw] = leaf_mask

            # Z-order: pasted leaf occludes everything underneath
            if masks_np.shape[0] > 0:
                masks_np = masks_np * (1 - full_mask[None, ...])
            if new_masks:
                for j in range(len(new_masks)):
                    new_masks[j] = new_masks[j] * (1 - full_mask)

            ys, xs = np.where(full_mask)
            if len(ys) == 0:
                continue
            box = [float(xs.min()), float(ys.min()),
                   float(xs.max() + 1), float(ys.max() + 1)]
            new_masks.append(full_mask)
            new_boxes.append(box)

        if not new_masks:
            # Nothing got pasted (no valid positions) — return original.
            return image, target

        # Stack all masks (existing + new) and rebuild target
        all_masks_np = np.concatenate(
            [masks_np, np.stack(new_masks, axis=0)], axis=0
        )
        # Drop existing masks that became fully empty after subtraction
        keep_old_idx = (
            existing_masks_keep := masks_np.reshape(masks_np.shape[0], -1).sum(axis=1) > 0
        ) if masks_np.shape[0] > 0 else np.zeros(0, dtype=bool)
        keep = np.concatenate(
            [keep_old_idx, np.ones(len(new_masks), dtype=bool)]
        )
        all_masks_np = all_masks_np[keep]

        # Rebuild boxes from the (now possibly clipped) masks
        rebuilt_boxes: list[list[float]] = []
        for i in range(all_masks_np.shape[0]):
            ys, xs = np.where(all_masks_np[i])
            if len(ys) == 0:
                rebuilt_boxes.append([0.0, 0.0, 1.0, 1.0])  # placeholder, will be sanitized
                continue
            rebuilt_boxes.append([
                float(xs.min()), float(ys.min()),
                float(xs.max() + 1), float(ys.max() + 1),
            ])

        new_image_tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        )
        new_image_tv = tv_tensors.Image(new_image_tensor)

        n_total = all_masks_np.shape[0]
        new_target = dict(target)
        new_target["masks"] = tv_tensors.Mask(
            torch.from_numpy(all_masks_np).to(torch.uint8)
        )
        new_target["boxes"] = tv_tensors.BoundingBoxes(
            torch.tensor(rebuilt_boxes, dtype=torch.float32),
            format="XYXY", canvas_size=(H, W),
        )
        new_target["labels"] = torch.ones((n_total,), dtype=torch.int64)
        new_target["area"] = torch.tensor(
            all_masks_np.reshape(n_total, -1).sum(axis=1),
            dtype=torch.float32,
        )
        new_target["iscrowd"] = torch.zeros((n_total,), dtype=torch.int64)

        return new_image_tv, new_target


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Simple dilation via scipy or a fall-back convolution. Returns
    binary uint8."""
    if px <= 0:
        return mask.astype(np.uint8)
    try:
        from scipy.ndimage import binary_dilation
        kernel_size = 2 * px + 1
        out = binary_dilation(
            mask.astype(bool),
            iterations=1,
            structure=np.ones((kernel_size, kernel_size), dtype=bool),
        )
        return out.astype(np.uint8)
    except ImportError:
        # Fallback: max-pool style dilation via PIL
        from PIL import ImageFilter
        pil = Image.fromarray((mask * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.MaxFilter(2 * px + 1))
        return (np.array(pil) > 0).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════
# CLI sanity check — `python -m final_project.augment.copy_paste`
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    """Quick smoke test on a single PACE val image."""
    import argparse
    from final_project.data.dataset import LettuceCOCODataset
    from final_project.data.transforms import (
        get_eval_transforms, WrapAsTVTensors, UnwrapFromTVTensors,
    )
    from torchvision.transforms import v2 as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--pots-dir", default="crops_full/pots")
    ap.add_argument("--bank-dir", default="leaf_bank")
    ap.add_argument("--out", default="results/copy_paste_demo.png")
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    # Eval pipeline + copy-paste appended for the smoke test
    pot_loader = make_pot_loader(args.coco, args.pots_dir)
    name_lookup = make_filename_lookup(args.coco)
    cp = CopyPaste(
        leaf_bank_dir=args.bank_dir,
        pot_mask_loader=pot_loader,
        filename_from_image_id=name_lookup,
        stage_match="strict",
        n_paste_range=(2, 4),
        p_apply=1.0,
        seed=42,
    )
    transform = T.Compose([WrapAsTVTensors(), cp, T.SanitizeBoundingBoxes(),
                           UnwrapFromTVTensors()])
    ds = LettuceCOCODataset(
        images_dir=args.images_dir,
        coco_path=args.coco,
        transforms=transform,
    )

    img_before_t, _ = LettuceCOCODataset(
        images_dir=args.images_dir,
        coco_path=args.coco,
        transforms=get_eval_transforms(),
    )[args.sample_idx]
    img_after_t, target_after = ds[args.sample_idx]

    before = (img_before_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    after = (img_after_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # Side-by-side: original | augmented | mask overlay
    H = max(before.shape[0], after.shape[0])
    W = before.shape[1] + after.shape[1] + after.shape[1] + 30
    grid = np.full((H, W, 3), 40, dtype=np.uint8)
    grid[: before.shape[0], : before.shape[1]] = before
    x = before.shape[1] + 15
    grid[: after.shape[0], x : x + after.shape[1]] = after

    # Mask overlay
    overlay = after.copy()
    masks = target_after["masks"].numpy()
    for i in range(masks.shape[0]):
        colour = np.random.RandomState(i + 7).randint(64, 255, size=3)
        overlay[masks[i] > 0] = (
            0.5 * overlay[masks[i] > 0] + 0.5 * colour
        ).astype(np.uint8)
    x2 = x + after.shape[1] + 15
    grid[: overlay.shape[0], x2 : x2 + overlay.shape[1]] = overlay

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"smoke demo → {args.out}")
    print(f"  before: {before.shape}  instances: 0+ (read from target)")
    print(f"  after:  {after.shape}   instances: {masks.shape[0]}")
