# final_project/

All code for the lettuce leaf segmentation project lives here. Run scripts
from the **project root** (one level up) so relative paths to
`Dataset/`, `crops_full/`, `annotations/`, etc. resolve correctly:

```bash
cd /Users/tharmmm/Documents/project
python final_project/<script>.py
```

## Folder layout

```
final_project/
├── (existing preprocessing scripts at root of this folder)
├── data/      — Dataset class, transforms, CVPPP → COCO conversion
├── models/    — model factories (Mask R-CNN, YOLOv8-seg, traditional)
├── train/     — training loops
├── eval/      — evaluation metrics + per-stage analysis
└── augment/   — copy-paste augmentation + leaf-bank building
```

## Existing scripts (preprocessing, already run)

| Script | Purpose |
|---|---|
| `inspect_dataset.py` | EDA on raw dataset → `eda_out/` |
| `crop_plants.py` | Crop per-plant images from tray photos → `crops_full/` |
| `split_dataset.py` | Train/test split (hold-out by tray) → `splits/` |
| `pick_val_set.py` | Pick stratified val subset → `splits/val.txt` |
| `prepare_cvat_upload.py` | Bundle crops as zip for CVAT upload |
| `sam_sanity_check.py` | SAM 2 auto-mode test (exploration only) |
| `sam_point_prompt.py` | SAM 2 point-prompted annotation helper |
| `visualize_annotations.py` | Render COCO labels overlaid on crops → `annotation_viz/` |

## Upcoming work — per subfolder

### `data/`
- `dataset.py` — `LettucePlantDataset` PyTorch class
- `transforms.py` — augmentation pipeline (flip, rotate, color, copy-paste)
- `cvppp_to_coco.py` — convert CVPPP `_label.png` → COCO JSON

### `models/`
- `traditional.py` — colour/plant-mask + watershed pipeline
- `mask_rcnn.py` — torchvision Mask R-CNN factory + classifier head swap
- `yolov8_seg.py` — ultralytics YOLOv8-seg wrapper

### `train/`
- `train_maskrcnn.py` — Mask R-CNN training loop
- `train_yolov8.py` — YOLOv8-seg training (ultralytics API)

### `eval/`
- `metrics.py` — AP50, IoU, MAE/RMSE implementations
- `evaluate.py` — generic evaluation driver (works for any method)
- `per_stage_analysis.py` — break metrics down by growth stage

### `augment/`
- `leaf_bank.py` — build leaf bank from hand-labelled crops
- `copy_paste.py` — augmentation function (uses pot-cell mask constraint)
