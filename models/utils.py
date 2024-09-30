import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import average_precision_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def compute_macro_fpr(targets, preds, num_classes):
    """
    Compute the False Positive Rate (FPR) for each class and the macro-average FPR.
    """
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + cm.sum(axis=1) - np.diag(cm) + np.diag(cm))
    with np.errstate(divide='ignore', invalid='ignore'):
        FPR = FP / (FP + TN)
        FPR = np.nan_to_num(FPR)  # Replace NaN with 0
    fpr_macro = np.mean(FPR)
    return fpr_macro, FPR



def precision_recall_curve_per_class(targets, y_scores, num_classes):
    pr_aucs = []
    for i in range(num_classes):
        binary_targets = (targets == i).astype(int)
        class_scores = y_scores[:, i]
        if binary_targets.sum() == 0:
            pr_auc = np.nan  # Handle classes not present in targets
        else:
            pr_auc = average_precision_score(binary_targets, class_scores)
        pr_aucs.append(pr_auc)
    return pr_aucs


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, reduction='mean'):
        """
        :param alpha: A scaling factor for class n.
        :param beta: Focusing parameter to control the degree of loss attenuation.
        :param class_weights: Optional tensor of class weights for handling class imbalance.
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.epsilon = 1e-12

    def forward(self, inputs, targets):
        """
        Forward pass for NID Loss.
        :param inputs: Predicted logits of shape [batch_size, num_classes].
        :param targets: Ground truth labels of shape [batch_size].
        """
        # Convert targets to one-hot encoding
        targets_one_hot = torch.eye(inputs.size(1), device=targets.device)[targets]
        
        # Apply softmax to the inputs to get probabilities
        probs = F.softmax(inputs, dim=1)
        # Get the probability for the correct class
        p_t = (probs * targets_one_hot).sum(dim=1)
        p_t = p_t.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        # Compute the NID loss as per the formula
        nid_loss = -self.alpha * (1 - p_t) ** self.beta * torch.log(p_t)
        
        # Apply the reduction method
        if self.reduction == 'mean':
            return nid_loss.mean()
        elif self.reduction == 'sum':
            return nid_loss.sum()
        else:
            return nid_loss


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, fixed_lr, prev_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.fixed_lr = fixed_lr
        self.prev_lr = prev_lr
        self.last_epoch = 0
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            return [self.fixed_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [self.prev_lr for _ in self.optimizer.param_groups]