# MGRL

MGRL is a multi-reward reinforcement learning framework for remote sensing change captioning on LEVIR-CC. It generates natural language descriptions of changes between paired images using cross-entropy training and self-critical sequence training (SCST).

The training objective combines CIDEr, BLEU, change-state, detail, and knowledge-graph rewards, with different rewards activated at different training stages. 

* **MGRL-RSCC: Multi-Granularity Reward Reinforcement Learning for Fine-Grained Remote Sensing Change Captioning**,
  Futian Wang, Mengqi Wang, Xiao Wang*, Wentao Wu, Haowen Wang, Zhicheng Zhao, Jin Tang,
  arXiv:2609.26166 [[Paper](https://arxiv.org/abs/2609.26166)]


Remote Sensing Change Captioning (RSCC), which aims to generate accurate and detailed linguistic descriptions of ground object variations from bi-temporal remote sensing images, is a critical and challenging task in intelligent remote sensing interpretation. The mainstream autoregressive training paradigm faces severe exposure bias and train-test distribution mismatch, resulting in cumulative generation errors. They tend to produce conservative and template-fixed captions while ignoring subtle scene change details. To address these challenges, this paper proposes a novel multi-granularity reward reinforcement learning paradigm, termed MGRL-RSCC. Specifically, we first leverage a CNN and hierarchical self-attention module to extract and enhance visual features from bi-temporal remote sensing images. A Transformer decoder is then utilized to complete visual-to-linguistic translation. Different from existing methods, we design a dual-decoding strategy and a two-stage joint optimization scheme, which combines token-level supervised learning via greedy decoding and multi-granularity reward-driven self-critical reinforcement learning via sampling decoding. We further construct three complementary reward functions covering linguistic fluency, change state consistency, and structural-semantic relevance to comprehensively optimize caption quality and alleviate false and missing change descriptions. Extensive experiments on multiple public RSCC benchmark datasets demonstrate that the proposed MGRL-RSCC effectively mitigates exposure bias and conservative generation problems in traditional autoregressive methods. 

<img width="1024" height="720" alt="image" src="https://github.com/user-attachments/assets/5f793bcd-7a6b-414e-8f48-a9e7b605d7f5" />




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


## Citation 

If you find this work useful for your research, please **cite** the following papers. 

```
@misc{wang2026MGRLRSCC,
      title={MGRL-RSCC: Multi-Granularity Reward Reinforcement Learning for Fine-Grained Remote Sensing Change Captioning}, 
      author={Futian Wang and Mengqi Wang and Xiao Wang and Wentao Wu and Haowen Wang and Zhicheng Zhao and Jin Tang},
      year={2026},
      eprint={2609.26166},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.26166}, 
}
```


