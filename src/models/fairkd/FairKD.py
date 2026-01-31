#!/usr/bin/env/python

import torch.nn as nn
import torch.nn.functional as F

from src.models.gcn.GCN import GCN


class FairKD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 p_dropout=0.0, temperature=4.0, lambda_kd=0.5):
        super(FairKD, self).__init__()

        self.teacher = GCN(input_size, hidden_size, output_size, p_dropout)
        self.student = GCN(input_size, hidden_size, output_size, p_dropout)

        self.temperature = temperature
        self.lambda_kd = lambda_kd

    def freeze_teacher(self):
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, adj, x, mode="student"):
        if mode == "teacher":
            return self.teacher(adj, x)
        return self.student(adj, x)

    def kd_loss(self, student_logits, teacher_logits):
        T = self.temperature
        return F.kl_div(F.log_softmax(student_logits / T, dim=1),
                        F.softmax(teacher_logits / T, dim=1),
                        reduction="batchmean") * (T * T)
