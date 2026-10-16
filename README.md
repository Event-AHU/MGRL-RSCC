# MGRL

MGRL is a multi-reward reinforcement learning framework for remote sensing change captioning on LEVIR-CC. It generates natural language descriptions of changes between paired images using cross-entropy training and self-critical sequence training (SCST).

The training objective combines CIDEr, BLEU, change-state, detail, and knowledge-graph rewards, with different rewards activated at different training stages.

## Setup

Prepare a Python environment with compatible PyTorch and torchvision installations, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

METEOR evaluation also requires Java and the following resources, which are not included:

```text
eval_func/meteor/meteor-1.5.jar
eval_func/meteor/data/paraphrase-en.gz
```

Alternatively, specify their locations with `JAVA_BIN`, `METEOR_JAR`, and `METEOR_PARAPHRASE`.

## Data Preparation

Organize the prepared dataset as follows:

```text
datasets/LEVIR_CC/
├── images/
│   ├── train/A/  train/B/
│   └── val/A/    val/B/
├── train.txt
├── val.txt
├── vocab.json
├── tokens/
└── levir_kg4_three_cleaned.csv
```

Training uses the `train` split, while validation and checkpoint selection use the `val` split. The knowledge-graph CSV must use GBK encoding. Datasets and model weights are not included in this repository.

## Training

```bash
bash run_mgrl.sh
```

The default configuration uses 150 epochs, a batch size of 32, top-k sampling with `k=3`, and a sampling temperature of `0.8`. SCST starts at epoch 8, with loss weights of `0.9` for cross-entropy and `0.1` for reinforcement learning. Epoch indices start at zero.

Outputs are saved to `checkpoints/levir_mgrl/`. To select a Python interpreter or GPU:

```bash
PYTHON_BIN=python3 GPU_ID=0 bash run_mgrl.sh
```

Custom paths can be supplied through `DATA_FOLDER`, `LIST_PATH`, `TOKEN_FOLDER`, `KG_CSV`, and `SAVEPATH`. Default paths are resolved relative to the project directory.

## Main Files

- `work2_train_unified_reward_lrboost_once.py`: multi-reward SCST training and validation.
- `model/`: image encoders and the caption decoder.
- `kg_reward_utils.py`: knowledge-graph processing and matching utilities.
- `run_mgrl.sh`: main experiment configuration.

## Notes

`selected_results.csv` contains historical experiment results, not a new evaluation of this release. Results may vary across runs. Saved checkpoints contain model weights, not the complete state required by the existing resume-training branch.

Upstream licenses and third-party attribution notices are preserved in `LICENSE.txt` and the evaluation modules.
