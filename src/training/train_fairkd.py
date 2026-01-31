#!/usr/bin/env/python

import time
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.metrics import f1_score, accuracy, fairness_metrics, auc_score
from src.utils.utils import format_metric
from src.utils.EarlyStoppingCriterion import EarlyStoppingCriterion
from src.models.fairkd.FairKD import FairKD
from src.paths import MODEL_DIR


def train_step(model, optimizer, adj, x, y, idx_train):
    model.train()
    optimizer.zero_grad()

    student_logits = model(adj, x, mode="student")
    with torch.no_grad():
        teacher_logits = model(adj, x, mode="teacher")

    loss_task = F.cross_entropy(student_logits[idx_train], y[idx_train])
    loss_kd = model.kd_loss(student_logits[idx_train], teacher_logits[idx_train])

    loss = (1 - model.lambda_kd) * loss_task + model.lambda_kd * loss_kd
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, adj, x, y, idx, sensitive=None):
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

    metrics = {"loss": loss, "acc": acc, "f1": f1, "auc": auc}

    # Calculate fairness if sensitive attribute is provided
    if sensitive is not None:
        sp, eo = fairness_metrics(predictions, y, sensitive, idx)
        metrics.update({"sp": sp, "eo": eo})

    return metrics


def train_fairkd(data, args):
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
    model.load_state_dict(torch.load(f'{model_save_basename}_best.pt'))

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
