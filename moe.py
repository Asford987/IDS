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

class ExpertModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, dropout_rate):
        super(ExpertModel, self).__init__()
        layers = []
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
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
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, num_experts))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.softmax(self.model(x), dim=1)


class MixtureOfExperts(pl.LightningModule):
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_units, 
                 gate_hidden_units, num_active_experts, dropout_rate, 
                 learning_rate=1e-2, class_weights=None):
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
        gating_weights = self.gate(x)
        topk_weights, topk_indices = torch.topk(gating_weights, self.top_k, dim=-1)
        gating_mask = torch.zeros_like(gating_weights).scatter_(-1, topk_indices, topk_weights)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.sum(expert_outputs * gating_mask.unsqueeze(2), dim=-1)
        return output

    def get_expert_activations(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_output = self.gate(x)
        return expert_outputs, gate_output
    
    def on_fit_start(self):
        torch.cuda.empty_cache()
        self.expert_usage_count = self.expert_usage_count.to(self.device)
        
    def on_train_epoch_start(self) -> None:
        self.criterion = nn.CrossEntropyLoss(self.class_weights.to(self.device))
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(y.cpu())

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, logger=True, prog_bar=True)
        return loss

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f2',
                'interval': 'epoch',
                'frequency': 1
            }
        }
                    

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
