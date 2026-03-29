#!/usr/bin/env python

import torch


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, index: torch.Tensor) -> float:
    """
    Computes the standard accuracy score for a subset of nodes.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels for all nodes.
    y_pred : torch.Tensor
        Model predictions (hard labels) for all nodes.
    index : torch.Tensor
        Mask or index array selecting the nodes to evaluate (e.g., test set).

    Returns
    -------
    float
        The fraction of correctly classified samples.
        Returns 0.0 if index is empty.
    """
    if len(y_true[index]) == 0:
        return 0.0
    return torch.mean((y_true[index] == y_pred[index]).float()).item()


def precision(y_true: torch.Tensor, y_pred: torch.Tensor, index: torch.Tensor) -> float:
    """
    Computes precision, averaged per-class (Macro) or for the positive class.

    For binary classification, returns the precision of the highest label index.
    For multi-class, returns the macro-averaged precision across all classes.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Hard model predictions.
    index : torch.Tensor
        Evaluation indices.

    Returns
    -------
    float
        Precision value in range [0, 1].
        Returns 0.0 if index is empty.
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


def recall(y_true: torch.Tensor, y_pred: torch.Tensor, index: torch.Tensor) -> float:
    """
    Computes recall, averaged per-class (Macro) or for the positive class.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Hard model predictions.
    index : torch.Tensor
        Evaluation indices.

    Returns
    -------
    float
        Recall (True Positive Rate) in range [0, 1].
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


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, index: torch.Tensor) -> float:
    """
    Computes the F1-score.

    Follows the same averaging logic as precision and recall
    (Macro-averaging for multi-class, positive-class only for binary).

    Parameters
    ----------
    y_true : torch.Tensor
        Ground truth labels.
    y_pred : torch.Tensor
        Hard model predictions.
    index : torch.Tensor
        Evaluation indices.

    Returns
    -------
    float
        F1-score in range [0, 1].
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


def fairness_metrics(y_pred: torch.Tensor, y_true: torch.Tensor,
                     sensitive: torch.Tensor,
                     index: torch.Tensor) -> tuple[list[float], list[float]]:
    """
    Calculates group fairness gaps: Statistical Parity and Equal Opportunity.

    The gaps are calculated as the difference between the advantaged group (1)
    and the disadvantaged group (0).
    A value of 0.0 indicates perfect parity.

    Parameters
    ----------
    y_pred : torch.Tensor
        Hard model predictions.
    y_true : torch.Tensor
        Ground truth labels.
    sensitive : torch.Tensor
        Binary sensitive attribute (1 = Advantaged, 0 = Disadvantaged).
    index : torch.Tensor
        Evaluation indices.

    Returns
    -------
    SP : list of float
        Statistical Parity Difference per class: $P(\hat{y}=i | s=1) - P(\hat{y}=i | s=0)$.
    EO : list of float
        Equal Opportunity Difference per class: $P(\hat{y}=i | y=i, s=1) - P(\hat{y}=i | y=i, s=0)$.
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


def auc_score(y_true: torch.Tensor, y_prob: torch.Tensor) -> float:
    """
    Computes Area Under the ROC Curve (AUC) for binary classification.

    This implementation uses the Wilcoxon-Mann-Whitney U-statistic approach,
    calculating the probability that a randomly chosen positive instance
    is ranked higher than a randomly chosen negative one.

    Parameters
    ----------
    y_true : torch.Tensor
        Binary ground truth labels.
    y_prob : torch.Tensor
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        AUC score.
        Returns 0.5 if only one class is present in y_true.
    """
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
