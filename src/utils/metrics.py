#!/usr/bin/env python
import torch


def accuracy(y_true, y_pred, index):
    """
    Accuracy of model predictions.

    Parameters
    ----------
    y_true: torch.Tensor
        true labels
    y_pred: torch.Tensor
        model predictions for all nodes
    index: torch.Tensor
        indices of nodes to evaluate (e.g., test set indexes)

    Returns
    -------
    accuracy: float
        accuracy
    """
    if len(y_true[index]) == 0:
        return 0.0
    return torch.mean((y_true[index] == y_pred[index]).float()).item()


def precision(y_true, y_pred, index):
    """
    Precision of model predictions averaged per class or of positive class in case of binary classification.

    Parameters
    ----------
    y_true: torch.Tensor
        true labels
    y_pred: torch.Tensor
        model predictions for all nodes
    index: torch.Tensor
        indices of nodes to evaluate (e.g., test set indexes)

    Returns
    -------
    precision: float
        precision
    """
    if len(y_true[index]) == 0:
        return 0.0

    y_t = y_true[index]
    y_p = y_pred[index]

    prec = []
    for label in range(y_true.max().item() + 1):
        true_pos = y_t == label
        true_neg = y_t != label

        pred_pos = y_p == label

        TP = (true_pos & pred_pos).sum().item()
        FP = (true_neg & pred_pos).sum().item()

        p = TP / (TP + FP) if TP + FP > 0 else 0.0
        prec.append(p)

    if len(y_true.unique()) == 2:
        return prec[-1]
    return torch.mean(torch.Tensor(prec)).item()


def recall(y_true, y_pred, index):
    """
    Recall of model predictions averaged per class or of positive class in case of binary classification.

    Parameters
    ----------
    y_true: torch.Tensor
        true labels
    y_pred: torch.Tensor
        model predictions for all nodes
    index: torch.Tensor
        indices of nodes to evaluate (e.g., test set indexes)

    Returns
    -------
    recall: float
        recall
    """
    if len(y_true[index]) == 0:
        return 0.0

    y_t = y_true[index]
    y_p = y_pred[index]

    rec = []
    for label in range(y_true.max().item() + 1):
        true_pos = torch.where(y_t == label)[0]

        pred_pos = torch.where(y_p == label)[0]
        pred_neg = torch.where(y_p != label)[0]

        TP = (true_pos & pred_pos).sum().item()
        FN = (true_pos & pred_neg).sum().item()

        r = TP / (TP + FN) if TP + FN > 0 else 0.0
        rec.append(r)

    if len(y_true.unique()) == 2:
        return rec[-1]
    return torch.mean(torch.Tensor(rec)).item()


def f1_score(y_true, y_pred, index):
    """
    F1-score of model predictions averaged per class or of positive class in case of binary classification.

    Parameters
    ----------
    y_true: torch.Tensor
        true labels
    y_pred: torch.Tensor
        model predictions for all nodes
    index: torch.Tensor
        indices of nodes to evaluate (e.g., test set indexes)

    Returns
    -------
    f1: float
        f1-score
    """
    if len(y_true[index]) == 0:
        return 0.0

    y_t = y_true[index]
    y_p = y_pred[index]

    F1 = []
    for label in range(y_true.max().item() + 1):
        true_pos = y_t == label
        true_neg = y_t != label

        pred_pos = y_p == label
        pred_neg = y_p != label

        TP = (true_pos & pred_pos).sum().item()
        FP = (true_neg & pred_pos).sum().item()
        FN = (true_pos & pred_neg).sum().item()

        denom = (2 * TP) + FP + FN

        f1 = (2 * TP) / denom if denom > 0 else 0.0
        F1.append(f1)

    if len(y_true.unique()) == 2:
        return F1[-1]
    return torch.mean(torch.Tensor(F1)).item()


def fairness_metrics(y_pred, y_true, sensitive, index):
    """
    Calculates Statistical Parity (SP) and Equal Opportunity (EO) for each class. These are fairness metrics.

    Parameters
    ----------
    y_true: torch.Tensor
        true labels
    y_pred: torch.Tensor
        model predictions for all nodes
    sensitive: torch.Tensor[1 | 0]
        specifying whether node is part of advantaged or disadvantaged group (1 or 0)
    index: torch.Tensor
        indices of nodes to evaluate (e.g., test set indexes)

    Returns
    -------
    SP, EO: Tuple[list[float], list[float]]
        SP: list of Statistical Parity values, one per class

        EO: list of Equal Opportunity values, one per class
    """
    SP = []
    EO = []

    y_p = y_pred[index]
    y_t = y_true[index]
    idx_adv = torch.where(sensitive[index] == 1)[0]
    idx_dis = torch.where(sensitive[index] == 0)[0]

    for label in range(y_true.max().item() + 1):
        # SP (Statistical Parity)
        # Difference in probability of being predicted as class i between groups.
        pred_i0 = (y_p[idx_dis] == label).sum().item()
        pred_i1 = (y_p[idx_adv] == label).sum().item()

        sp = (pred_i1 / idx_adv.shape[0]) - (pred_i0 / idx_dis.shape[0])
        SP.append(sp)

        # EO (Equal Opportunity)
        # Difference in true positive rates for class i between groups.
        p_y0 = torch.where(y_t[idx_dis] == label)[0]
        p_y1 = torch.where(y_t[idx_adv] == label)[0]

        if p_y0.numel() == 0 or p_y1.shape.numel() == 0:
            eo = 0
        else:
            p_iy0 = (y_p[idx_dis][p_y0] == label).sum().item()
            p_iy1 = (y_p[idx_adv][p_y1] == label).sum().item()
            eo = (p_iy1 / p_y1.shape[0]) - (p_iy0 / p_y0.shape[0])
        EO.append(eo)

    return SP, EO


def auc_score(y_true, y_prob):
    y_true = y_true.view(-1)
    y_prob = y_prob.view(-1)

    desc_score_indices = torch.argsort(y_prob, descending=True)
    y_true_sorted = y_true[desc_score_indices]

    cum_tp = torch.cumsum(y_true_sorted, dim=0)

    total_pos = cum_tp[-1]
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tp_at_neg_indices = cum_tp[y_true_sorted == 0]

    auc = tp_at_neg_indices.sum() / (total_pos * total_neg)

    return auc.item()
