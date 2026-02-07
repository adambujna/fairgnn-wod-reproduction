# fairGNN-WOD: Fair Graph Learning Without Demographics

This repository is dedicated to a reproduction of the paper ***fairGNN-WOD:
Fair Graph Learning Without Demographics*** by Wang, Liu, Pan, Liu, Saeed, Qiu & Zhang
(2025).
It was created for the **FACT: Fairness, Accountability, Correctness,
and Transparency** course at the **University of Amsterdam
(UvA)**.
The goal is to reproduce the results of the paper,
which presents a solution for achieving fairness in Graph Neural Networks
(GNNs) without the need for complete demographic information.

## Overview

Graph Neural Networks (GNNs) can potentially perform unfairly
(e.g., different prediction accuracy or true positive rate) across different demographic groups.
Because of this, many efforts have been made to ensure more fair results for different groups.
Due to privacy or legal reasons, the procurement of demographic information is not always feasible,
which complicates attempts to achieve fair results.

*fairGNN-WOD: Fair Graph Learning Without Demographics (Wang, et al., 2025)* 
proposes a method of analyzing and mitigating societal bias in GNNs without access to demographic information.
In addition,
the paper also puts forward techniques at expressing the trade-off between prediction accuracy and fairness.
On several benchmark datasets the proposed methods perform comparably in prediction accuracy
while achieving a higher level of fairness than baselines.

## Requirements

The project is designed to likely also work with their newer versions but was completed with the following libraries:
```toml
numpy=1.26
torch=2.2
torch-geometric=2.7
torchdata=0.11
```

For viewing or running the notebooks in `notebook`, these packages are required:
```toml
notebook = [
  "jupyter",
  "notebook",
  "matplotlib"
]
```

To read the optional legacy dgl version data we used:
```toml
dgl=2.2
```

## Project Structure
```
.
├── checkpoints/                # Best checkpoints of each model. fGNN-WOD split into `stage1` (VGAE) and `stage2`. Stage-2 training looks for VGAE in `checkpoints/stage1`.
│   └── download_checkpoints.sh # Bash script to download checkpoints.
├── data/                       # Location of all datasets (`.pt` or `.bin` files).
│   └── download_datasets.sh/   # Bash script to download datasets.
├── notebooks/              # Jupyter notebooks for data pre-processing and visualizations.
├── outputs/                # Training logs of each model. fGNN-WOD split into `stage1` (VGAE) and `stage2`.
├── src/                    # --- Project source code ---
│   ├── __init__.py
│   ├── dataset/            # Custom dataset class for graph datasets in the correct format.
│   ├── models/             # - Model implementations -
│   │   ├── fairgnn_wod/        # Second stage of fGNN-WOD and its components.
│   │   ├── fairkd/             # FairKD baseline, uses GCN modules.
│   │   ├── gcn/                # GCN baseline and Graph-Convolution modules.
│   │   └── vgae/               # Demographic VGAE and HGR estimator networks.
│   ├── training/           # Training scrtips for each model which are called by `src/main.py`. fGNN-WOD split into `stage1` (VGAE) and `stage2`.
│   ├── utils/              # Utility modules and functions.
│   ├── main.py             # Main script. Run with arguments to run model training.
│   └── paths.py            # Global path references.
├── LICENSE             # License
├── README.md
└── pyproject.toml      # Dependencies
```

## Usage

1. **Clone the repository and install dependencies**
```shell
git clone https://github.com/adambujna/fairgnn-wod-reproduction.git
cd fairgnn-wod-reproduction
pip install -e .
```
2. **Download data and (optionally) model checkpoints**

Datasets:
```shell
cd data
download_datasets.sh
```
Model checkpoints:
```shell
cd checkpoints
sh download_checkpoints.sh
```
3. **Run experiments**

All tasks are executed by the main script.
```shell
python -m src.main --model fairgnnwod --device cuda --task train --dataset pokec_z --epochs 200 --lr 0.001 --hidden 64 --seed 0
```

## Hyperparameters
Some notable hyperparameters are listed here.

* `--lr`: learning rate of model optimizer.
* `--weight_decay`: L2 regularization strength.
* `--epochs`: number of training epochs.
* `--hidden`: hidden embedding size for GCN and fairGNN-WOD models.
* `--dropout`: dropout rate.
* `--patience`: number of epochs to wait before early stopping.
* `--warmup`: number of warmup epochs before enabling early stopping. Determines loss introduction schedule.

VGAE-specific (Stage 1) hyperparameters:

* `--hidden_vgae`: hidden layer size in the VGAE encoder.
* `--latent_size`: dimensionality of the latent embeddings.
* `--hgr_steps_per_epoch`: number of HGR estimation net training steps per epoch.
* `--lambda_hgr`: weight for the HGR regularization term.
* `--hidden_hgr`: hidden size for the HGR discriminator network.

fairGNN-WOD–specific (Stage 2) hyperparameters:

* `--channels`: number of latent factor channels in the disentangled representation.
* `--lambda_dd`: how strongly the discriminator loss propagates to the rest of the network and modifies it.
* `--alpha`: weight for the channel independence (*LI*) loss.
* `--beta`: weight for the discriminator demographic (*LD*) loss.
* `--gamma`: weight for the fairness (*LF*) loss.
* `--adv_steps_per_epoch`: number of adversarial demographic discriminator training steps per epoch.
* `--mask_warmup`: fraction of epochs to warm up before applying masking.
  Allows discriminator to learn on unmasked data early.

FairKD-specific hyperparameters:

* `--kd_temp`: temperature parameter for FairKD.
* `--kd_lambda`: weight for the distillation loss.


## Paper Citation

If you use this repository in your work, please cite the original paper:

Wang, Z., Liu, F., Pan, S., Liu, J., Saeed, F., Qiu, M., & Zhang, 
W. fairGNN-WOD: Fair Graph Learning Without Demographics.
In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25 (pp. 556–564).
```
@inproceedings{ijcai2025p63,
  title     = {fairGNN-WOD: Fair Graph Learning Without Complete Demographics},
  author    = {Wang, Zichong and Liu, Fang and Pan, Shimei and Liu, Jun and Saeed, Fahad and Qiu, Meikang and Zhang, Wenbin},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {556--564},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/63},
  url       = {https://doi.org/10.24963/ijcai.2025/63},
}
```


---


## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- The authors of *fairGNN-WOD: Fair Graph Learning Without Demographics* for their work on the fairGNN-WOD framework.

- The University of Amsterdam for providing the course and platform for this project.
