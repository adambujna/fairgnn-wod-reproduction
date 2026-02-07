# Model Training

To train the models, use `main.py`.
It handles data loading, model initialization, and training loops for 
the four different architectures (GCN, VGAE, fairGNN-WOD, and FairKD).

## Hyperparameters

Below is a list of the settings and hyperparameters available via command-line arguments.

### Training Configuration

* `--model`: Choose model: `gcn`, `fairkd`, `vgae` (stage 1), `fairgnn-wod` (stage 2).
* `--task`: `train` or `predict`. Currently only training is implemented and tests instead if a checkpoint is already present.
* `--dataset`: Choose dataset: `credit`, `dblp`, `pokec_z`, `pokec_n`.
* `--dgl_data`: Whether to use dgl `.bin` versions of data what we were originally provided.
* `--device`: Pytorch device: `cpu` or `cuda`.
* `--seed`: Manual seed for reproducibility.
* `--print_every`: How often are evaluation and training time printed (in epochs).
* `--save_every`: How often is the model checkpoint saved (in epochs).

### General Hyperparameters

* `--epochs`: Maximum number of training epochs.
* `--patience`: Number of epochs with no improvement on the validation set to wait before early stopping.
* `--warmup`: Number of warmup epochs before enabling early stopping and before full loss scheduling takes effect.
* `--lr`: Learning rate for the model optimizer.
* `--weight_decay`: L2 regularization strength (weight decay).
* `--hidden`: Hidden embedding size for GCN and fairGNN-WOD and classification layers.
* `--dropout`: Dropout probability applied to linear layers.

### VGAE-Specific (Stage 1)

These parameters control the Variational Graph Autoencoder used to infer sensitive attributes.

* `--hidden_vgae`: Hidden dimension size in the VGAE encoder GCN layers.
* `--latent_size`: Dimensionality of the latent embeddings *Z*.
* `--hgr_steps_per_epoch`: Number of training steps the HGR estimator takes for every one step the VGAE takes.
* `--lambda_hgr`: Weight coefficient for the HGR maximal correlation regularization term.
* `--hidden_hgr`: Hidden dimension size for the HGR discriminator network.

### fairGNN-WOD–Specific (Stage 2)

These parameters control the main debiasing framework.

* `--channels`: Number of latent factor channels in the disentangled representation.
* `--lambda_dd`: Weight controlling how strongly the discriminator's gradient affects the encoder (controls the Gradient Reversal Layer strength).
* `--alpha`: Weight for the Channel Independence (*LI*) loss.
* `--beta`: Weight for the Discriminator Demographic (*LD*) loss.
* `--gamma`: Weight for the Fairness (*LF*) loss.
* `--adv_steps_per_epoch`: Number of steps the demographic discriminator trains for every one step of the main model.
* `--mask_warmup`: Fraction of total epochs (`0.0` to `1.0`) to wait before applying masking to the sensitive attributes. This allows the discriminator to learn on unmasked data at the start of training.

### FairKD-Specific

* `--kd_temp`: Temperature parameter for Knowledge Distillation (softens the teacher/student logits).
* `--kd_lambda`: Weighting factor balancing the distillation loss vs. the task loss.

---

## Training Workflow of the fairGNN-WOD

The fairGNN-WOD framework trains in two stages.
The VGAE of Stage 1 must be pre-trained before training Stage 2, as 
Stage 2 relies on the VGAE to generate pseudo-labels for sensitive attributes.

### Step 1: Pre-train the VGAE

First, the VGAE learns to reconstruct the graph structure and node features through correlations with the sensitive 
attribute.

```shell
python -m src.main --model vgae --dataset pokec_z --hgr_steps_per_epoch 5 --lambda_hgr 0.5
```

This will save a `.pt` model file (e.g., `checkpoints/stage1/vgae_pokec_z_best.pt`) which is required for the next 
step. 
*It will also save the last weights of the HGR estimator (e.g., `checkpoints/stage1/hgr_pokec_z_last.pt`), 
but these are not further necessary.*

### Step 2: Train FairGNN-WOD

Once the VGAE is trained, it is frozen and the second stage of fairGNN-WOD is trained. 
The VGAE is used to infer demographic quasi-labels for the fairness losses of stage 2. 
Care to match the VGAE hyperparameter settings to properly load the pre-trained VGAE. 

```shell
python -m src.main --model fairgnnwod --dataset pokec_z --alpha 1.0 --beta 1.0 --gamma 5.0 --hidden_hgr 512 --hidden_vgae 256 --latent_size 32
```

---

## Loss Scheduling

Training for some models uses a **Loss Scheduler** (by user only controllable by `--warmup` and `--mask_warmup`) to 
stabilize training.

**Why is this used?**
In adversarial and multi-objective learning (like fairGNN-WOD and VAEs), the multiple loss terms often compete.
If all constraints are applied with full force at Epoch 0, the model may struggle to learn the primary classification 
task, leading to mode collapse or random guessing.

**How it works:**

1. **General Warmup**: For the first `--warmup` epochs, the weights for auxiliary losses (like *LI*) are typically 
   set to 0 or a very low value.
   They are linearly ramped up to their target values after the warmup period ends. 
   This allows the model to establish high classification accuracy first before being "corrected" for fairness.
2. **Mask Warmup**: Specific to fairGNN-WOD, the model only actually starts masking sensitive information after the 
   proportion of epochs specified in `--mask_warmup` have elapsed (e.g., if `--mask_warmup 0.2`, the model will 
   begin masking after 1/5 of epochs have elapsed).
   This way, the discriminator is allowed
   to train on fully visible embeddings at the start of training to become a strong adversary
   before the encoder attempts to fool it with disentangled and masked embeddings.

### VGAE (Stage 1) ###
* **KL loss**: The KL loss of the ELBO is gradually increased to `0.1` with a sigmoid schedule after a short warm-up 
  of `0.1 * --warmup` epochs to allow the model to learn reconstruction before prior is enforced.

* `lambda_hgr` is linearly increased during warm-up.

### FairGNN-WOD (Stage 2)

* **Independence loss (*LI*)**: The weight `--alpha` is linearly increased starting at `0.4 * --warmup` until the 
  end of warm-up, encouraging channel independence only after initial representation learning.

* **Demographic classification loss (*LD*)**: The weight `--beta` is linearly increased during warmup starting at 
  `--warmup * --mask_warmup`,
  allowing the model to first learn stable embeddings before learning to separate sensitive and 
  demographic-irrelevant channels.

* **Fairness loss (*LF*)**: The weight `--gamma` is linearly increased from `0.2 * --warmup` to `0.6 * --warmup` to 
  allow the model to learn stable representations first before enforcing fairness during training.

* `lambda_dd` is increased with a sigmoid schedule from `0.6 * --warmup` to the end of warm-up to smoothly introduce 
  the discriminator gradients to the encoder.

---

## Loss Functions Components

>*For details about the losses see the original formulations of these models.*

This section details how hyperparameters control the weight of each loss.

The hyperparameter in the parentheses scales the component of the loss next to it.

### FairGNN-WOD Loss

The total objective function for the second stage is a weighted sum of four components:

*L* = *LP* + *LI* + *LD* + *LF*

* ***LP (unscaled)***: The standard Cross-Entropy classification loss.
* ***LI (`--alpha`)***: Enforces orthogonality between the disentangled channels.
* ***LD (`--beta`)***: The adversarial loss from the demographic discriminator.
  Maximizing this forces the encoder 
  to hide demographic information.
* ***LF (`--gamma`)***: Explicit fairness regularization (correlation reduction between Y and S) on the predictions.
* **`--lambda_dd`**: Scaling factor for the backwards gradient coming from the discriminator.
  It determines how much the discriminator's accuracy influences the encoder.

### FairKD Loss

FairKD balances the task loss and the distillation loss:

*L* = *LP* + *L_KL*

* ***LP (1 - `--kd_lambda`)***: Standard Cross-Entropy against true labels.
* ***L_KL (`--kd_lambda`)***: KL-Divergence between the Student's soft logits and Teacher's soft logits.
* ***`--kd_temp`***: Applied inside the softmax function to smooth the probability distributions.

### VGAE Loss (Stage 1)

The VGAE loss is a sum of two elements:

*L* = ELBO + *L_HGR*

* **ELBO**: Reconstruction loss and prior regularization.
* ***L_HGR (`--lambda_hgr`)***: Maximizes correlation between the latent space *Z* and labels *Y*. This ensures the 
  inference of the latent sensitive attribute *S* is independent of the label *Y*.
