import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import fbeta_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class NIDLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, reduction='mean'):
        """
        :param alpha: A scaling factor for class n.
        :param beta: Focusing parameter to control the degree of loss attenuation.
        :param class_weights: Optional tensor of class weights for handling class imbalance.
        :param reduction: Specifies the reduction to apply to the output ('none', 'mean', 'sum').
        """
        super(NIDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

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
    def __init__(self, input_dim, num_experts, hidden_units, dropout_rate, class_weights, learning_rate=1e-3):
        super(GateModel, self).__init__()
        self.input_dim = input_dim
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, num_experts))
        self.model = nn.Sequential(*layers)
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.ones(self.num_classes)
        self.initialize_weights()
        self.save_hyperparameters()

    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)
    
    def on_fit_start(self):
        self.criterion = NIDLoss()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long().squeeze()
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())
        total_loss = self.criterion(y_hat, y)
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

        self.log('val_precision', prec, logger=True)
        self.log('val_recall', recall, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
        )
        
        # scheduler = WarmupScheduler(optimizer, warmup_epochs=5, fixed_lr=1e-1, prev_lr=self.learning_rate)
        
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


class ExpertModel(nn.Module):
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