# TADA

## Task-Agnostic Incremental Vision-Language Object Detection via Prompt Augmentation and Distribution-Aware Fusion

This repository contains the PyTorch implementation of [TADA](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/4881.pdf), a framework for task-agnostic incremental vision-language object detection (TA-IVLOD).

Incremental vision-language object detection adapts a pretrained open-vocabulary detector to a sequence of visual tasks without forgetting previously learned concepts. In the task-agnostic setting, task identities are unavailable at inference time, so the detector must recognize classes from all learned tasks simultaneously.

TADA addresses this setting with two complementary components:

- **Stochastic Prompt Augmentation** injects noise into textual prompts during training to reduce cross-task semantic interference.
- **Test-Time Distribution-Aware Fusion** dynamically weights class-specific LoRA experts at inference time to resolve conflicts between task-specific adaptations.

The implementation uses Grounding DINO with LoRA adaptation and evaluates sequential learning on **ODinW-13**. It also reports zero-shot generalization on **MS COCO**.

## Setup

### Prerequisites

- Python 3.9
- A CUDA-capable PyTorch installation
- CUDA toolkit and a compatible C++ compiler for the Deformable-DETR operators

### Environment

Create and activate a Python environment, then install the Python dependencies:

```bash
conda create -n tada python=3.9 pip
conda activate tada

pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6"

git clone https://github.com/fundamentalvision/Deformable-DETR.git
cd Deformable-DETR/models/ops
sh make.sh
python test.py
cd ../../..
```

### Pretrained weights

Download the Grounding DINO checkpoint used to initialize the detector:

```bash
mkdir -p weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -O weights/groundingdino_swint_ogc.pth
```

The model configuration also expects a local BERT text encoder at `weights/bert-base-uncased`:

```bash
huggingface-cli download google-bert/bert-base-uncased \
  --local-dir weights/bert-base-uncased
```

## Datasets

Place the datasets under `datasets/` before running an experiment.

- **ODinW-13** is used for sequential incremental learning. The data loaders expect each subset under `datasets/odinw13/<dataset-name>/` with its original `train` and `test` annotations. For example, the AerialMaritimeDrone configuration reads from `datasets/odinw13/AerialMaritimeDrone/tiled/`.
- **MS COCO 2017 validation** is used for zero-shot evaluation.

The ODinW-13 benchmark in this repository contains the following task subsets: AerialMaritimeDrone, Aquarium, CottontailRabbits, EgoHands, NorthAmericaMushrooms, Packages, PascalVOC, pistols, pothole, Raccoon, ShellfishOpenImages, thermalDogsAndPeople, and VehiclesOpenImages.

## Running TADA

The provided experiment script uses the Grounding DINO checkpoint, the ODinW-13 configuration, and a shuffled task order:

```bash
bash train_tada.sh
```

Its equivalent command is:

```bash
python -u main.py \
  --config-file test/test_odinw13 \
  --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_tada.py \
  --model-checkpoint-path weights/groundingdino_swint_ogc.pth \
  --output-dir output/tada \
  --seed 0 \
  --num-gpus 1
```

### Evaluate a trained model

Use `--eval-only` to skip sequential training and evaluate an existing experiment output:

```bash
python -u main.py \
  --config-file test/test_odinw13 \
  --model-config-file groundingdino/config/GroundingDINO_SwinT_OGC_dt_tada.py \
  --model-checkpoint-path weights/groundingdino_swint_ogc.pth \
  --output-dir output/tada \
  --num-gpus 1 \
  --eval-only
```

The specified output directory must contain `last_lora.pth` and the LDA statistics in `features/cov_matrix_stream.pt` and `features/class_mean_stream.pt`, which are produced by sequential training.

### Useful options

| Option | Description |
| --- | --- |
| `--model-config-file` | Grounding DINO model configuration. |
| `--model-checkpoint-path` | Initial Grounding DINO checkpoint. |
| `--output-dir` | Directory for checkpoints, feature statistics, logs, and evaluation results. |
| `--num-gpus` | Number of GPUs used by Detectron2's launcher. |
| `--eval-only` | Skip sequential training and evaluate the model saved in `--output-dir`. |
| `--shot` | Use the 1-, 5-, or 10-shot ODinW-13 configuration. |
| `--lora-r` | LoRA rank. |
| `--lora-alpha` | LoRA scaling coefficient. |
| `--lora-lr` | Learning rate for LoRA parameters. |

## Citation

```bibtex
@inproceedings{jiang2026task,
  title={Task-Agnostic Incremental Vision-Language Object Detection via Prompt Augmentation and Distribution-Aware Fusion},
  author={Jiang, Yonghan and Xie, Zhengyuan and Liu, Wenchu and Huang, Linlan and Yang, Fei and Liu, Xialei},
  booktitle={European Conference on Computer Vision},
  pages={172--190},
  year={2026},
  organization={Springer}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).

## Acknowledgements

This codebase builds on [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), and [DitHub](https://github.com/chiara-cap/DitHub). Please cite the corresponding work when using their code or models.
