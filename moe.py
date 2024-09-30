import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import fbeta_score, precision_score, recall_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm


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

class GateModel(pl.LightningModule):
    def __init__(self, input_dim, num_experts, hidden_units, dropout_rate, class_weights=None, learning_rate=1e-3, alpha=1.0, beta=1.0):
        super(GateModel, self).__init__()
        self.input_dim = input_dim
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.Mish())
            input_dim = units
        layers.append(nn.Linear(input_dim, num_experts))
        self.model = nn.Sequential(*layers)
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.ones(self.num_experts)
        self.initialize_weights()
        self.save_hyperparameters()

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def on_fit_start(self):
        self.criterion = FocalLoss(self.alpha, self.beta)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long().squeeze()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_logits = []  
        self.val_targets = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
        preds = torch.argmax(F.softmax(y_hat, dim=1), dim=1)
        
        self.val_preds.append(preds.cpu())
        self.val_logits.append(y_hat.cpu())
        self.val_targets.append(y.cpu())
        
        total_loss = self.criterion(y_hat, y)
        self.log('val_total_loss', total_loss, logger=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        
        y_true = label_binarize(targets.numpy(), classes=list(range(self.num_experts)))  # One-hot encoding for multiclass
        logits = torch.cat(self.val_logits)  # Assuming you're storing the raw outputs during validation
        y_scores = F.softmax(logits, dim=1).cpu().numpy()

        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        f2_score_macro = fbeta_score(targets_np, preds_np, beta=2, average='macro', zero_division=1)
        precision_macro = precision_score(targets_np, preds_np, average='macro', zero_division=1)
        recall_macro = recall_score(targets_np, preds_np, average='macro', zero_division=1)
        pr_auc_macro = average_precision_score(y_true, y_scores, average='macro')
        
        fpr_macro, fpr_per_class = compute_macro_fpr(targets_np, preds_np, self.num_experts)
        

        self.log('val_precision', precision_macro, logger=True)
        self.log('val_recall', recall_macro, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
        self.log('val_fpr', fpr_macro, logger=True)
        self.log('val_pr_auc', pr_auc_macro, logger=True)
        
        f2_scores = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average=None, zero_division=1)
        precision_scores = precision_score(targets.numpy(), preds.numpy(), average=None, zero_division=1)
        recall_scores = recall_score(targets.numpy(), preds.numpy(), average=None, zero_division=1)
        pr_auc_scores = precision_recall_curve_per_class(targets.numpy(), y_scores, self.num_experts)
        
        for class_idx, (f2_score, prec_score_class, recall_score_class, pr_auc_score_class, fpr_class) in enumerate(
            zip(f2_scores, precision_scores, recall_scores, pr_auc_scores, fpr_per_class)
        ):
            self.log(f'val_f2_class_{class_idx}', f2_score, logger=True, prog_bar=False)
            self.log(f'val_precision_class_{class_idx}', prec_score_class, logger=True, prog_bar=False)
            self.log(f'val_recall_class_{class_idx}', recall_score_class, logger=True, prog_bar=False)
            self.log(f'val_fpr_class_{class_idx}', fpr_class, logger=True, prog_bar=False)
            self.log(f'val_pr_auc_class_{class_idx}', pr_auc_score_class, logger=True, prog_bar=False)

            
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
        )
        
        # scheduler = WarmupScheduler(optimizer, warmup_epochs=2, fixed_lr=1e-1, prev_lr=self.learning_rate)
        
        schedulers = [
            {
                'scheduler': scheduler1,
                'monitor': 'val_f2',
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'ReduceLROnPlateau'
            },
            # {
            #     'scheduler': scheduler,
            #     'interval': 'epoch',
            #     'frequency': 1,
            #     'name': 'CustomLRScheduler'
            # }
        ]

        return [optimizer], schedulers


class ExpertModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super(ExpertModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    def on_fit_start(self): pass
    def training_step(self, batch, batch_idx): pass
    def on_validation_step_start(self): pass
    def validation_step(self, batch, batch_idx): pass
    def on_validation_epoch_end(self) -> None: pass
    def on_before_optimizer_step(self, optimizer) -> None: pass
    def configure_optimizers(self): pass

class MoE(pl.LightningModule):
    def __init__(self, gate_model: GateModel, num_expert_models, expert_hidden_units, learning_rate=1e-3, dropout_rate=0.2, class_weights=None):
        super(MoE, self).__init__()
        self.save_hyperparameters()

        self.gate = gate_model.to(self.device)
        self.experts = nn.ModuleList([ExpertModel(self.gate.input_dim, 1, expert_hidden_units, dropout_rate) for _ in range(num_expert_models)])
        self.num_classes = num_expert_models
        self.learning_rate = learning_rate
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.ones(self.num_classes)
        self.criterion = nn.CrossEntropyLoss(self.class_weights.to(self.device))

    def forward(self, x):
        gating_weights = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  
        output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=-1)  
        return output, gating_weights

    def load_balancing_loss(self, gating_weights, y):
        expert_utilization = torch.mean(gating_weights, dim=0)
        class_weights = self.class_weights.to('cuda')[y]
        weighted_utilization = torch.mean(gating_weights * class_weights.unsqueeze(1), dim=0)
        ideal_utilization = torch.ones_like(expert_utilization) / self.num_classes
        loss = torch.norm(weighted_utilization - ideal_utilization)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long().squeeze()
        outputs, weights = self(x)
        preds = torch.argmax(outputs, dim=1)
        self.train_preds.append(preds.detach().cpu())
        self.train_targets.append(y.detach().cpu())
        total_loss = self.criterion(outputs, y) + self.load_balancing_loss(weights, y)
        
        self.log('train_loss', total_loss, logger=True, prog_bar=True)
        return total_loss

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_targets = []
        self.criterion = nn.CrossEntropyLoss(self.class_weights.to(self.device))

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds)
        targets = torch.cat(self.train_targets)
        f2_score_macro = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average='macro')
        prec = precision_score(targets.numpy(), preds.numpy(), average='macro', zero_division=1)
        recall = recall_score(targets.numpy(), preds.numpy(), average='macro', zero_division=1)
        self.log('train_precision', prec, logger=True)
        self.log('train_recall', recall, logger=True)
        self.log('train_f2', f2_score_macro, prog_bar=True, logger=True)
        self.train_preds = []
        self.train_targets = []


    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat, weights = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        total_loss = self.criterion(y_hat, y) + self.load_balancing_loss(weights, y)
        self.log('val_total_loss', total_loss, logger=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self.val_preds = []
        self.val_targets = []

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        
        f2_score_macro = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average='macro')
        prec = precision_score(targets.numpy(), preds.numpy(), average='macro')
        recall = recall_score(targets.numpy(), preds.numpy(), average='macro')

        f2_scores = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average=None, zero_division=1)
        f2_scores = torch.tensor(f2_scores, dtype=torch.float32).to(self.device)
        for class_idx, f2_score in enumerate(f2_scores):
            self.log(f'val_f2_class_{class_idx}', f2_score, logger=True, prog_bar=False)

        self.class_weights = self.compute_class_weights(f2_scores)

        self.log('val_precision', prec, logger=True)
        self.log('val_recall', recall, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
        )
        
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, fixed_lr=1e-2, prev_lr=self.learning_rate)
        
        schedulers = [
            {
                'scheduler': scheduler1,
                'monitor': 'val_f2',
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'ReduceLROnPlateau'
            },
            {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'CustomLRScheduler'
            }
        ]

        return [optimizer], schedulers
        
    def compute_class_weights(self, f2_scores, smoothing_factor=0.5, epsilon=1e-8, min_weight=0.1, max_weight=10.0):
            """
            Compute class weights based on F2 scores with smoothing and clipping.

            Parameters:
            - f2_scores: A tensor of per-class F2 scores.
            - smoothing_factor: The power to which the F2 score denominator is raised (e.g., 0.5 for sqrt).
            - epsilon: Small value to avoid division by zero.
            - min_weight: Minimum weight value to clip to.
            - max_weight: Maximum weight value to clip to.

            Returns:
            - class_weights: A tensor of computed class weights.
            """
            # Convert f2_scores to a PyTorch tensor if not already
            if not isinstance(f2_scores, torch.Tensor):
                f2_scores = torch.tensor(f2_scores, dtype=torch.float32)

            # Apply smoothing to the inverse F2 scores
            smoothed_weights = 1 / ((f2_scores + epsilon) ** smoothing_factor)

            # Clip the weights to avoid extreme values
            class_weights = torch.clamp(smoothed_weights, min=min_weight, max=max_weight)

            return class_weights