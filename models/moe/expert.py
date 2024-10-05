import os

from models.utils import FocalLoss, compute_macro_fpr, precision_recall_curve_per_class
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
import numpy as np
import pandas as pd



class ExpertPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, path, binarize_on_label):
        if isinstance(path, str): 
            if path.endswith('.csv'): data = pd.read_csv(path)
            elif path.endswith('.parquet'): data = pd.read_parquet(path)
        else: data = path
        
        self.X = data.drop(columns=['Label']).astype(np.float64)
        self.y = np.asarray(data['Label'] == binarize_on_label, dtype=np.int64) 
    
    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y[idx]
    
    def __len__(self):
        return len(self.X)


class ExpertPretrainDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, binarize_on_label, batch_size=4096):
        super(ExpertPretrainDataModule, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.binarize_on_label = binarize_on_label
        
    def setup(self, stage=None):
        self.train_dataset = ExpertPretrainDataset(self.train_path, self.binarize_on_label)
        self.test_dataset = ExpertPretrainDataset(self.test_path, self.binarize_on_label)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False)


class ExpertModel(pl.LightningModule):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.0, learning_rate=1e-3):
        super(ExpertModel, self).__init__()
        self.input_dim = input_dim
        layers = []
        for units in hidden_units:
            layers.append(nn.BatchNorm1d(input_dim, dtype=torch.float64))
            layers.append(nn.Linear(input_dim, units, dtype=torch.float64))
            layers.append(nn.LeakyReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, 1, dtype=torch.float64))
        self.model = nn.Sequential(*layers)
        self.num_experts = 1
        self.learning_rate = learning_rate
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
        self.criterion = FocalLoss()

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
        preds = (torch.sigmoid(y_hat) >= 0.5).long()
        self.val_preds.append(preds.cpu())
        self.val_logits.append(y_hat.cpu())
        self.val_targets.append(y.cpu())
        
        total_loss = self.criterion(y_hat, y)
        self.log('val_loss', total_loss, logger=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)
        
        y_true = label_binarize(targets.numpy(), classes=[0,1])
        logits = torch.cat(self.val_logits)
        y_scores = F.sigmoid(logits).cpu().numpy()
        
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        
        cm = confusion_matrix(targets_np, preds_np, labels=[0, 1])
        tp, fp, fn, tn = cm.ravel()
        
        self.log('val_fp', fp, prog_bar=True, logger=True)
        self.log('val_fn', fn, prog_bar=True, logger=True)
        self.log('val_tp', tp, logger=True)
        self.log('val_tn', tn, logger=True)
        
        f2_score_macro = fbeta_score(targets_np, preds_np, beta=2, average='binary', zero_division='warn')
        precision_macro = precision_score(targets_np, preds_np, average='binary', zero_division='warn')
        recall_macro = recall_score(targets_np, preds_np, average='binary', zero_division='warn')
        pr_auc_macro = average_precision_score(y_true, y_scores, average='macro')
        
        fpr_macro, _ = compute_macro_fpr(targets_np, preds_np, self.num_experts)
        
        self.log('val_precision', precision_macro, logger=True)
        self.log('val_recall', recall_macro, logger=True)
        self.log('val_f2', f2_score_macro, prog_bar=True, logger=True)
        self.log('val_fpr', fpr_macro, logger=True)
        self.log('val_pr_auc', pr_auc_macro, logger=True)
        
    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
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