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
import numpy as np
import pandas as pd



class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        data = pd.read_parquet(path)
        self.X = data.drop(columns=['Label']).astype(np.float32)
        self.y = np.asarray(data['Label'].astype('category').cat.codes, dtype=np.int64) 
    
    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y[idx]
    
    def __len__(self):
        return len(self.X)


class AutoEncoderDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, batch_size=4096):
        super(AutoEncoderDataModule, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = AutoEncoderDataset(self.train_path)
        self.test_dataset = AutoEncoderDataset(self.test_path)
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False)


class AutoEncoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_units, h_dim, dropout_rate=0.0, learning_rate=1e-3):
        super(AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.learning_rate = learning_rate
        self.initialize_weights()
        self.save_hyperparameters()
        
    def create_encoder(self):
        layers = []
        input_dim = self.input_dim
        for units in self.hidden_units:
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, self.h_dim))
        return nn.Sequential(*layers)
        
    def create_decoder(self):
        layers = []
        input_dim = self.h_dim
        for units in reversed(self.hidden_units):
            layers.append(nn.BatchNorm1d(input_dim))
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.LeakyReLU())
            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(self.dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, self.input_dim))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def encode(self, x) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(self, x) -> torch.Tensor:
        return self.decoder(self.encoder(x))
    
    def on_fit_start(self):
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        loss = self.criterion(logits, x)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            y_hat = self(x)
        
        total_loss = self.criterion(y_hat, x)
        self.log('val_loss', total_loss, logger=True, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
        )
        
        schedulers = [
            {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'reduce_on_plateau': True,
                'interval': 'epoch',
                'frequency': 1,
                'name': 'ReduceLROnPlateau'
            },
        ]

        return [optimizer], schedulers