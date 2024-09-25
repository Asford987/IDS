import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import fbeta_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class WarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, warmup_epochs, fixed_lr, prev_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.fixed_lr = fixed_lr
        self.prev_lr = prev_lr
        self.last_epoch = 0
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            # Return fixed learning rate during warmup period
            return [self.fixed_lr for _ in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            # After warmup, return the learning rate according to the optimizer's initial values
            return [self.prev_lr for _ in self.optimizer.param_groups]

class TemperatureScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, mode='max', patience=5, increase_factor=1.1, increase_duration=5,
                 min_lr=1e-6, max_lr=1.0, verbose=False):
        # Initialize the parent class with factor=1.0 to prevent automatic LR reduction
        super().__init__(optimizer, mode=mode, patience=patience, factor=0.9, min_lr=min_lr, verbose=verbose)
        self.patience = patience
        self.increase_factor = increase_factor
        self.duration = increase_duration
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.verbose = verbose
        self.wait = 0
        self.prev_metric = None
        self.increased_lr = False
        self.increase_counter = 0
        self.original_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, metrics=None, epoch=None):
        current_metric = float(metrics)
        
        if self.prev_metric is None:
            # First epoch
            self.prev_metric = current_metric
            self.wait = 0
        else:
            if current_metric > self.prev_metric + 1e-8:
                # Improvement observed
                self.prev_metric = current_metric
                self.wait = 0
            else:
                # No improvement
                self.wait += 1

            if self.increased_lr:
                self.increase_counter += 1
                if self.increase_counter >= self.duration:
                    # Reset learning rate after duration
                    self._reset_lr(epoch)
                    self.increased_lr = False
                    self.increase_counter = 0
            elif self.wait >= self.patience:
                # Increase learning rate
                self._increase_lr(epoch)
                self.increased_lr = True
                self.increase_counter = 0
                self.wait = 0  # Reset wait after increasing LR

    def _increase_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = param_group['lr']
            new_lr = min(old_lr * self.increase_factor, self.max_lr)
            param_group['lr'] = new_lr
            if self.verbose:
                print(f'Epoch {epoch}: increasing learning rate of group {i} to {new_lr:.4e}.')

    def _reset_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            new_lr = self.original_lrs[i]
            param_group['lr'] = new_lr
            if self.verbose:
                print(f'Epoch {epoch}: resetting learning rate of group {i} to {new_lr:.4e}.')



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

class GateModel(pl.LightningModule):
    def __init__(self, input_dim, num_experts, hidden_units, dropout_rate, class_weights, learning_rate=1e-2):
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
        self.criterion = nn.CrossEntropyLoss(self.class_weights)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)
    
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

        # Compute per-class F2 scores
        f2_scores = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average=None, zero_division=1)
        f2_scores = torch.tensor(f2_scores, dtype=torch.float32).to(self.device)

        # Log metrics
        self.log('val_precision', prec, logger=True)
        self.log('val_recall', recall, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=10, verbose=True, min_lr=1e-6
        )
        
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, fixed_lr=1e-1, prev_lr=self.learning_rate)
        
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
    

class MoE(pl.LightningModule):
    def __init__(self, gate_model: GateModel, num_expert_models, expert_hidden_units, learning_rate=5e-3, dropout_rate=0.2, class_weights=None):
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
        class_weights = self.class_weights.to('cuda')[y]  # Get weights corresponding to the batch's true labels
        weighted_utilization = torch.mean(gating_weights * class_weights.unsqueeze(1), dim=0)
        ideal_utilization = torch.ones_like(expert_utilization) / self.num_classes
        loss = torch.norm(weighted_utilization - ideal_utilization)
        return loss


    def on_fit_start(self):
        torch.cuda.empty_cache()
        
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


        # Log metrics
        self.log('val_precision', prec, logger=True)
        self.log('val_recall', recall, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=1e-2)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=10, verbose=True, min_lr=1e-6
        )
        
        scheduler = WarmupScheduler(optimizer, warmup_epochs=5, fixed_lr=1e-1, prev_lr=self.learning_rate)
        
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