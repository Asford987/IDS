import os

import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import fbeta_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch

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

    def step(self, metrics, epoch=None):
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

class GateModel(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_units, dropout_rate):
        super(GateModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, num_experts))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class MixtureOfExperts(pl.LightningModule):
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_units, 
                 gate_hidden_units, num_active_experts, dropout_rate, 
                 learning_rate=1e-1, class_weights=None):
        super(MixtureOfExperts, self).__init__()
        self.save_hyperparameters()

        self.experts = nn.ModuleList([ExpertModel(input_dim, output_dim, expert_hidden_units, dropout_rate) for _ in range(num_experts)])
        self.gate = GateModel(input_dim, num_experts, gate_hidden_units, dropout_rate)
        self.top_k = num_active_experts
        self.expert_usage_count = torch.zeros(num_experts, dtype=torch.float32)
        self.num_classes = num_experts
        self.learning_rate = learning_rate
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else torch.ones(self.num_classes)
        self.criterion = nn.CrossEntropyLoss(self.class_weights.to(self.device))

    def forward(self, x):
        gating_weights = self.gate(x)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)  # [batch_size, output_dim, num_experts]
        output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=-1)  # [batch_size, output_dim]
        return output, gating_weights, expert_outputs


    def get_expert_activations(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_output = self.gate(x)
        return expert_outputs, gate_output
    
    def on_fit_start(self):
        torch.cuda.empty_cache()
        self.expert_usage_count = self.expert_usage_count.to(self.device)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long().squeeze()
        outputs, _, expert_outputs = self(x)
        preds = torch.argmax(outputs, dim=1)
        # Collect predictions and targets
        self.train_preds.append(preds.detach().cpu())
        self.train_targets.append(y.detach().cpu())
        # Compute the loss
        total_loss, ce_loss, similarity_loss = self.criterion(outputs, y, expert_outputs)
        # Log losses
        self.log('train_loss', total_loss, logger=True, prog_bar=True)
        self.log('ce_loss', ce_loss, logger=True)
        self.log('similarity_loss', similarity_loss, logger=True)
        return total_loss

    def on_train_epoch_start(self):
        self.train_preds = []
        self.train_targets = []

    def on_train_epoch_end(self):
        preds = torch.cat(self.train_preds)
        targets = torch.cat(self.train_targets)
        f2_score_macro = fbeta_score(targets.numpy(), preds.numpy(), beta=2, average='macro')
        prec = precision_score(targets.numpy(), preds.numpy(), average='macro', zero_division=1)
        recall = recall_score(targets.numpy(), preds.numpy(), average='macro', zero_division=1)
        # Log metrics
        self.log('train_precision', prec, logger=True)
        self.log('train_recall', recall, logger=True)
        self.log('train_f2', f2_score_macro, prog_bar=True, logger=True)
        # Clear lists
        self.train_preds = []
        self.train_targets = []


    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat, _, experts_outputs = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

        total_loss, _, similarity_loss = self.criterion(y_hat, y, experts_outputs)
        self.log('val_total_loss', total_loss, logger=True, prog_bar=True)
        self.log('val_similarity_loss', similarity_loss, logger=True)
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
            optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
        )
        
        scheduler = TemperatureScheduler(optimizer, mode='max', patience=5, increase_factor=2.5,
                                         increase_duration=5, min_lr=1e-6, max_lr=1.0)
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
                'monitor': 'val_f2',
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'CustomLRScheduler'
            }
        ]

        return [optimizer], schedulers
                    

class ExpertUsageLogger(pl.Callback):
    def __init__(self, moe_model):
        super(ExpertUsageLogger, self).__init__()
        self.moe_model = moe_model
        self.expert_usage_history = []

    def on_train_epoch_end(self, trainer, pl_module):
        usage_count = self.moe_model.expert_usage_count.clone().cpu().numpy()
        self.expert_usage_history.append(usage_count)

    def plot_expert_usage(self):
        import matplotlib.pyplot as plt
        usage_history = torch.tensor(self.expert_usage_history)
        plt.figure(figsize=(10, 6))
        for i in range(usage_history.shape[1]):
            plt.plot(usage_history[:, i], label=f'Expert {i}')
        plt.xlabel('Epoch')
        plt.ylabel('Expert Usage Count')
        plt.title('Expert Usage Over Epochs')
        plt.legend(loc='upper left')
        plt.show()
