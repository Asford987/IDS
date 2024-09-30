import os

from models.utils import FocalLoss, compute_macro_fpr, precision_recall_curve_per_class
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import fbeta_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm



class FeedForward(pl.LightningModule):
    def __init__(self, input_dim, num_experts, hidden_units, dropout_rate=0.0, learning_rate=1e-3, alpha=1.0, beta=1.0):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.Mish())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, num_experts))
        self.model = nn.Sequential(*layers)
        self.num_experts = num_experts
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
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
        
        y_true = label_binarize(targets.numpy(), classes=list(range(self.num_experts)))
        logits = torch.cat(self.val_logits)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.8, patience=5, verbose=True, min_lr=1e-6
        )
        
        schedulers = [
            {
                'scheduler': scheduler,
                'monitor': 'val_f2',
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'ReduceLROnPlateau'
            },
        ]

        return [optimizer], schedulers