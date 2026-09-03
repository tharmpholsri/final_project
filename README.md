# Lettuce Leaf Instance Segmentation and Counting

This project segments and counts visible lettuce leaves in overhead images.
It compares three approaches:

1. a traditional edge-aware watershed method;
2. Mask R-CNN;
3. a pixel-embedding model that groups pixels into leaf instances.

The repository includes the source code and JHI leaf annotations. The image
datasets, trained checkpoints, leaf bank and full model outputs are not
included because of their size.

## Directory structure

- `annotations/` contains the JHI training, validation and test annotations.
- `data/` contains the COCO dataset loader and image transformations.
- `models/` contains the three model implementations.
- `train/` contains the training scripts for Mask R-CNN and pixel embedding.
- `eval/` contains evaluation scripts and the shared metrics.
- `augment/` contains the copy-paste augmentation used during training.
- `crop_plants.py` creates individual plant crops from the original tray images
  and masks.

The traditional method does not have a training script. Its parameters are
selected on the validation set by the evaluation script.

## Data

The project used two sources of data:

- CVPPP leaf segmentation data (Ara2012, Ara2013-Canon and Tobacco) for the
  initial training set. The instance labels were converted to COCO format.
- The JHI lettuce image sequence supplied for the project. Individual
  plants were cropped from tray images and a subset was manually annotated in
  COCO format. Separate tray groups were used for training, validation and
  testing to avoid image overlap between the splits.

The scripts expect the following layout relative to the directory from which
the commands are run:

```text
annotations/
  cvppp_coco.json
  instances_train.json
  instances_validation.json
  instances_test_set.json
crops_full/
  images/
  masks/
  pots/
leaf_bank/
Plant_Phenotyping_Datasets/
```

`crops_full/masks` contains plant foreground masks. `crops_full/pots` contains
pot-region masks used to restrict copy-paste placement. `leaf_bank` contains
the leaf cutouts used by the augmentation.

## Setup

The code was written in Python and uses PyTorch. The main dependencies are:

```text
torch
torchvision
numpy
Pillow
pycocotools
opencv-python
scipy
scikit-image
scikit-learn
```

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/tharmpholsri/final_project.git
cd final_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The source imports the repository as the `final_project` package. When running
commands from the repository root, add its parent directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

The commands below show how the main experiments were run. File paths may need
to be adjusted to match the local dataset and checkpoint locations.

## Data preparation

The JHI plant crops and pot masks were produced with:

```bash
python -m final_project.crop_plants \
  --root Dataset/lettuce_PACE176 \
  --out crops_full \
  --with-pots
```

CVPPP instance labels were converted to COCO format with:

```bash
python -m final_project.data.cvppp_to_coco \
  --cvppp-root Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant \
  --out annotations/cvppp_coco.json
```

The leaf bank used by copy-paste augmentation can be rebuilt from the JHI
training annotations:

```bash
python -m final_project.augment.build_leaf_bank \
  --coco annotations/instances_train.json \
  --images-dir crops_full/images \
  --out leaf_bank
```

## Mask R-CNN

The COCO zero-shot baseline is evaluated without replacing or training the
prediction heads. The validation sweep and fixed test run are:

```bash
python -m final_project.eval.evaluate_maskrcnn_zeroshot \
  --coco annotations/instances_validation.json \
  --images-dir crops_full/images \
  --sweep \
  --out results/maskrcnn_zeroshot_val.json

python -m final_project.eval.evaluate_maskrcnn_zeroshot \
  --coco annotations/instances_test_set.json \
  --images-dir crops_full/images \
  --mode all-classes \
  --score-threshold 0.05 \
  --out results/maskrcnn_zeroshot_test.json
```

The main Mask R-CNN experiment used CVPPP and the annotated JHI training
subset, with copy-paste augmentation:

```bash
python -m final_project.train.train_maskrcnn \
  --train-coco annotations/cvppp_coco.json \
  --train-images Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant \
  --train-coco-extra annotations/instances_train.json \
  --train-images-extra crops_full/images \
  --val-coco annotations/instances_validation.json \
  --val-images crops_full/images \
  --copy-paste \
  --leaf-bank leaf_bank \
  --pots-dir crops_full/pots \
  --epochs 25 \
  --batch-size 2 \
  --num-workers 4 \
  --ckpt-name maskrcnn_v2_c
```

Evaluation is run with the saved checkpoint. The score threshold should be
selected on validation data before running the final test evaluation:

```bash
python -m final_project.eval.evaluate_maskrcnn \
  --checkpoint checkpoints/maskrcnn_v2_c_best.pth \
  --coco annotations/instances_validation.json \
  --images-dir crops_full/images \
  --sweep \
  --out results/maskrcnn_v2_c_val.json

python -m final_project.eval.evaluate_maskrcnn \
  --checkpoint checkpoints/maskrcnn_v2_c_best.pth \
  --coco annotations/instances_test_set.json \
  --images-dir crops_full/images \
  --score-threshold 0.7 \
  --out results/maskrcnn_v2_c_test.json
```

## Pixel embedding

The final pixel-embedding experiment used the FPN H/2 decoder, one-dimensional
tags and an MSE foreground loss:

```bash
python -m final_project.train.train_pixel_embed \
  --train-coco annotations/cvppp_coco.json \
  --train-images Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant \
  --train-coco-extra annotations/instances_train.json \
  --train-images-extra crops_full/images \
  --val-coco annotations/instances_validation.json \
  --val-images crops_full/images \
  --architecture fpn_h2 \
  --embedding-dim 1 \
  --detection-loss mse \
  --epochs 25 \
  --batch-size 1 \
  --num-workers 4 \
  --ckpt-name pixel_embed_fpn_h2_paperlike
```

The paper-like decoder can be tuned on validation data and then applied to the
test set:

```bash
python -m final_project.eval.evaluate_pixel_embed_paperlike \
  --checkpoint checkpoints/pixel_embed_fpn_h2_paperlike_best.pth \
  --architecture fpn_h2 \
  --coco annotations/instances_validation.json \
  --images-dir crops_full/images \
  --plant-mask-dir crops_full/masks \
  --sweep \
  --sweep-csv results/pixel_embed_val_sweep.csv \
  --out results/pixel_embed_val.json

python -m final_project.eval.evaluate_pixel_embed_paperlike \
  --checkpoint checkpoints/pixel_embed_fpn_h2_paperlike_best.pth \
  --architecture fpn_h2 \
  --coco annotations/instances_test_set.json \
  --images-dir crops_full/images \
  --plant-mask-dir crops_full/masks \
  --hist-bins 128 \
  --peak-prominence 0.01 \
  --out results/pixel_embed_test.json
```

## Traditional method

The three watershed evaluators are `evaluate_traditional`,
`evaluate_traditional_v2` and `evaluate_traditional_v2_adaptive`. They
correspond to the distance-transform, image-boundary and adaptive methods.
The adaptive method uses separate parameter sets for smaller and larger
plants. Parameters are selected using the validation set:

```bash
python -m final_project.eval.evaluate_traditional_v2_adaptive \
  --coco annotations/instances_validation.json \
  --images-dir crops_full/images \
  --masks-dir crops_full/masks \
  --sweep \
  --out results/traditional_v2_adaptive_val.json
```

This creates `traditional_v2_adaptive_best_params.json` in the output
directory. The selected parameters can then be used on the test set:

```bash
python -m final_project.eval.evaluate_traditional_v2_adaptive \
  --coco annotations/instances_test_set.json \
  --images-dir crops_full/images \
  --masks-dir crops_full/masks \
  --best results/traditional_v2_adaptive_best_params.json \
  --out results/traditional_v2_adaptive_test.json
```

Each evaluation script writes COCO-format predictions and a JSON file
containing the overall and per-growth-stage metrics. The reported metrics
include AP, AP50, AP75, leaf-count MAE, RMSE and signed count error.

## Temporal analysis

Temporal processing uses predictions from the final Mask R-CNN model. The
following commands reproduce the dense inference, count smoothing and mask
tracking stages:

```bash
python -m final_project.eval.temporal_dense_inference
python -m final_project.eval.temporal_smoothing_potid

python -m final_project.eval.temporal_measure_rotation
python -m final_project.eval.temporal_dense_masks
python -m final_project.eval.temporal_track_sweep
python -m final_project.eval.temporal_track
```

These scripts expect the final checkpoint at
`checkpoints/maskrcnn_v2_c_best.pth`. Count smoothing and tracking parameters
are selected using validation MAE. The test values are reported only after the
validation setting has been fixed.
