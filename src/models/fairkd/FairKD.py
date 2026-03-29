#!/usr/bin/env/python

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gcn.GCN import GCN


class FairKD(nn.Module):
    """
    Fair Knowledge Distillation (FairKD) framework using GCN modules.

    This model implements a teacher-student architecture where a 'student'
    GCN learns to mimic the softened output distributions of a 'teacher' GCN.
    In the context of fairness, this is often used to distill knowledge from
    a fair teacher into a more compact or specialized student.

    Parameters
    ----------
    input_size : int
        Number of input features per node.
    hidden_size : int
        Hidden dimension for both teacher and student GCNs.
    output_size : int
        Number of output classes.
    p_dropout : float, optional
        Dropout probability. Defaults to 0.0.
    temperature : float, optional
        Scaling factor $T$ to soften the logits during distillation.
        Higher values create a flatter probability distribution. Defaults to 4.0.
    lambda_kd : float, optional
        Weighting coefficient for the distillation loss component. Defaults to 0.5.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 p_dropout: float = 0.0,
                 temperature: float = 4.0,
                 lambda_kd: float = 0.5):
        super(FairKD, self).__init__()

        self.teacher = GCN(input_size, hidden_size, output_size, p_dropout)
        self.student = GCN(input_size, hidden_size, output_size, p_dropout)

        self.temperature = temperature
        self.lambda_kd = lambda_kd

    def freeze_teacher(self) -> None:
        """
        Disables gradient computation and sets the teacher model to evaluation mode.

        This is called before the distillation phase to ensure only the student's weights are updated.
        """
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, adj: torch.Tensor, x: torch.Tensor, mode: str = "student") -> torch.Tensor:
        """
        Forward pass for either the teacher or student GCN.

        Parameters
        ----------
        adj : torch.Tensor
            Adjacency matrix [N, N].
        x : torch.Tensor
            Input node features [N, input_size].
        mode : str, optional
            Switch between "student" and "teacher" forward passes.
            Defaults to "student".

        Returns
        -------
        torch.Tensor
            Classification logits from the selected sub-model.
        """
        if mode == "teacher":
            return self.teacher(adj, x)
        return self.student(adj, x)

    def kd_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the Knowledge Distillation loss using KL-Divergence.

        Parameters
        ----------
        student_logits : torch.Tensor
            Logits produced by the student model.
        teacher_logits : torch.Tensor
            Logits produced by the teacher model.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the distillation loss.
        """
        T = self.temperature
        return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                        F.softmax(teacher_logits / T, dim=1),
                        reduction="batchmean") * (T * T)
