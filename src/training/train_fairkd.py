#!/usr/bin/env/python

import time
import os
import numpy as np
import torch
import torch.nn.functional as F

from typing import Any

from src.utils.metrics import f1_score, accuracy, fairness_metrics, auc_score
from src.utils.utils import format_metric
from src.utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from src.models.fairkd.FairKD import FairKD
from src.paths import MODEL_DIR


def train_step(
    model: FairKD,
    optimizer: torch.optim.Optimizer,
    adj: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx_train: torch.Tensor
) -> float:
    """
    Perform a single training iteration for the Student model using Knowledge Distillation.

    The student is trained to minimize a weighted combination of:
    1. **Task Loss**: Standard Cross-Entropy against the ground truth labels.
    2. **KD Loss**: Divergence between the Student's soft predictions and the
       pre-trained Teacher's soft predictions.

    Parameters
    ----------
    model : FairKD
        The distillation wrapper containing both the Teacher (frozen) and Student.
    optimizer : torch.optim.Optimizer
        Optimizer for the Student model parameters.
    adj : torch.Tensor
        Adjacency matrix [N, N].
    x : torch.Tensor
        Node feature matrix [N, D].
    y : torch.Tensor
        Target labels [N].
    idx_train : torch.Tensor
        Indices for the training set.

    Returns
    -------
    float
        The total combined loss value for the step.
    """
    model.train()
    optimizer.zero_grad()

    # Student pass
    student_logits = model(adj, x, mode="student")

    # Teacher pass
    with torch.no_grad():
        teacher_logits = model(adj, x, mode="teacher")

    # Compute loss components
    loss_task = F.cross_entropy(student_logits[idx_train], y[idx_train])
    loss_kd = model.kd_loss(student_logits[idx_train], teacher_logits[idx_train])

    # Final weighted loss: (1-lambda)*Task + lambda*KD
    loss = (1 - model.lambda_kd) * loss_task + model.lambda_kd * loss_kd
    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    adj: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    sensitive: torch.Tensor | None = None
) -> dict[str, float]:
    """
    Evaluate the Student model's performance and fairness metrics.

    Parameters
    ----------
    model : FairKD
        The model to evaluate.
    adj : torch.Tensor
        Adjacency matrix [N, N].
    x : torch.Tensor
        Node feature matrix [N, D].
    y : torch.Tensor
        Target labels [N].
    idx : torch.Tensor
        Indices for the evaluation split (val or test).
    sensitive : torch.Tensor, optional
        Ground truth sensitive attributes for calculating fairness metrics.

    Returns
    -------
    dict[str, float]
        Dictionary containing 'loss', 'acc', 'f1', 'auc', and fairness scores.
    """
    model.eval()
    output = model(adj, x)

    # Calculate basic metrics
    loss = F.cross_entropy(output[idx], y[idx]).item()
    predictions = output.argmax(dim=-1).long()
    acc = accuracy(predictions, y, idx)
    f1 = f1_score(predictions, y, idx)

    # AUC
    probs = F.softmax(output, dim=-1)[:, 1].detach().cpu()
    y_cpu = y.detach().cpu()
    idx_cpu = idx.detach().cpu()
    try:
        auc = auc_score(y_cpu[idx_cpu], probs[idx_cpu])
    except ValueError:
        auc = 0.5

    metrics: dict[str, Any] = {"loss": loss, "acc": acc, "f1": f1, "auc": auc}

    # Calculate fairness if sensitive attribute is provided
    if sensitive is not None:
        sp, eo = fairness_metrics(predictions, y, sensitive, idx)
        metrics.update({"sp": sp, "eo": eo})

    return metrics


def train_fairkd(data: Any, args: Any) -> None:
    """
    Orchestrate the Knowledge Distillation training process.

    This function:
    1. Prepares features (with conditional normalization for the Credit dataset).
    2. Loads a pre-trained Teacher model (usually a standard GCN).
    3. Freezes the Teacher to ensure only the Student learns.
    4. Executes the training loop with Early Stopping based on Validation F1.
    5. Performs final testing on the best-saved checkpoint.

    Parameters
    ----------
    data : GraphDataset
        Data object containing graph structure and features.
    args : argparse.args
        Configuration object containing hyperparameters (lr, kd_lambda, etc.).
    """
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    adj = data.get_adj_matrix(norm='both').to(device)
    x = data.x.to(device)
    if data.dataset == 'credit':    # It performs better with normalization
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
    y = data.y.to(device).long()

    idx_train = data.train_idx.to(device)
    idx_val = data.val_idx.to(device)
    idx_test = data.test_idx.to(device)

    # Handle Sensitive Attribute (if exists)
    sensitive = data.sensitive.to(device) if data.sensitive is not None else None

    # Initialize Model
    model = FairKD(
        input_size=data.x.shape[1],
        hidden_size=args.hidden,
        output_size=data.num_classes,
        p_dropout=args.dropout,
        temperature=args.kd_temp,
        lambda_kd=args.kd_lambda
    ).to(device)

    # Load pretrained teacher
    teacher_ckpt = f"{MODEL_DIR}/gcn/gcn_{args.dataset}_best.pt"
    model.teacher.load_state_dict(
        torch.load(teacher_ckpt, map_location=device)
    )
    model.freeze_teacher()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    model_save_basename = MODEL_DIR + f"/fairkd/fairkd_{args.dataset}"
    best_model_path = f'{model_save_basename}_best.pt'
    if os.path.exists(best_model_path):
        print(f"Found existing best model at {best_model_path}. Skipping training.")
    else:
        # Training Loop
        model_save_basename = MODEL_DIR + f"/fairkd/fairkd_{args.dataset}"
        if args.save_every != -1:    # save initialized weights at epoch 0 also if saving periodically
            torch.save(model.state_dict(), model_save_basename + '_0.pt')
        early_stop = EarlyStoppingCriterion(patience=args.patience,
                                            mode='max',
                                            start_from_epoch=args.warmup,
                                            save_basename=model_save_basename)
        print(f"Starting FairKD training on {args.dataset}...")
        t_total = time.time()

        for epoch in range(args.epochs):
            t = time.time()
            loss_train = train_step(model, optimizer, adj, x, y, idx_train)

            # Validate
            val_metrics = evaluate(model, adj, x, y, idx_val)

            if args.print_every != -1 and (epoch + 1) % args.print_every == 0:
                print(f"Epoch: {epoch + 1} | "
                      f"Loss: {loss_train:.4f} | "
                      f"Val F1: {val_metrics['f1']} | "
                      f"Val Acc: {val_metrics['acc']:.4f} | "
                      f"Time: {time.time() - t:.4f}s")

            if args.save_every != -1 and (epoch + 1) % args.save_every == 0:
                torch.save(model.state_dict(), f"/{model_save_basename}_{epoch + 1}.pt")

            if early_stop.step(val_metrics['f1'], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Training Finished. Total Time: {time.time() - t_total:.4f}s")

    # Testing
    print("\nRestoring best model...")
    model.load_state_dict(torch.load(best_model_path))

    test_metrics = evaluate(model, adj, x, y, idx_test, sensitive)

    print("--------------------------------------------------")
    print(f"Test Set Results ({args.dataset}):")
    print(f"     AUC: {test_metrics['auc']:.4f}")
    print(f"Accuracy: {test_metrics['acc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")

    if sensitive is not None:
        print("Fairness Metrics:")
        for k, v in test_metrics.items():
            if k not in ['loss', 'acc', 'f1', 'auc']:
                print(f"  {k}: {format_metric(v, precision=4)}")
    print("--------------------------------------------------")
