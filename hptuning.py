import os

import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from metaflow import FlowSpec, step, Parameter, trigger_on_finish
import numpy as np
from moe import *
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset



@trigger_on_finish(flow='PreprocessingFlow')
class HyperparameterTuningFlow(FlowSpec):

    artifact_storage = Parameter('artifact_storage', default='artifacts/clean_data')
    n_trials = Parameter('n_trials', default=20)

    @step
    def start(self):
        self.X_train = pd.read_csv(self.artifact_storage + '/X_train.csv').astype(np.float32)
        self.X_test = pd.read_csv(self.artifact_storage + '/X_test.csv').astype(np.float32)
        self.y_train = np.asarray(np.load(self.artifact_storage + '/y_train.npy'), np.float32)
        self.y_test = np.asarray(np.load(self.artifact_storage + '/y_test.npy'), np.float32)
        self.input_dim = self.X_train.shape[1]
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        self.output_dim = 1
        self.next(self.tuning)

    @step
    def tuning(self):
        def objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim):
            gate_hidden_units_options = {
                "16": [16], 
                "32": [32], 
                "64": [64], 
                "32_16": [32, 16]
            }
            
            chosen_gate_hidden_units_str = trial.suggest_categorical('gate_hidden_units', list(gate_hidden_units_options.keys()))
            chosen_gate_hidden_units = gate_hidden_units_options[chosen_gate_hidden_units_str]

            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            
            model = MixtureOfExperts(
                input_dim=input_dim,
                output_dim=output_dim,
                num_experts=output_dim,
                expert_hidden_units=[32, 64, 32],
                gate_hidden_units=chosen_gate_hidden_units,
                num_active_experts=3,
                dropout_rate=dropout_rate
            )
            
            logger = TensorBoardLogger("logs", name="MoE_experimental")
            csv_logger = pl.loggers.CSVLogger("logs", name="MoE_experimental")
            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            trainer = pl.Trainer(
                max_epochs=300,
                logger=[logger, csv_logger],
                callbacks=[lr_monitor],
                accelerator='gpu',
            )
            
            train_loader = DataLoader(TensorDataset(torch.tensor(X_train.values, device='cuda'), torch.tensor(y_train, device='cuda')), batch_size=4096, shuffle=True)
            val_loader = DataLoader(TensorDataset(torch.tensor(X_val.values, device='cuda'), torch.tensor(y_val, device='cuda')), batch_size=4096)
            
            trainer.fit(model, train_loader, val_loader)
            
            val_f2 = trainer.callback_metrics["val_f2"].item()

            return val_f2
        
        gate_hidden_units_options = {
            "16": [16], 
            "32": [32], 
            "64": [64], 
            "32_16": [32, 16]
        }
        study = optuna.create_study(direction="maximize")
        
        study.optimize(lambda trial: objective(trial, self.X_train, self.y_train, self.X_test, self.y_test, self.input_dim, self.output_dim), 
                    n_trials=self.n_trials)
        
        print(f"Best Hyperparameters: {study.best_params}")
        
        best_params = study.best_params
        best_gate_hidden_units = gate_hidden_units_options[best_params['gate_hidden_units']]
        
        best_model = MixtureOfExperts(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_experts=self.output_dim, 
            expert_hidden_units=[32, 64, 32],
            gate_hidden_units=best_gate_hidden_units,
            num_active_experts=3,
            dropout_rate=best_params['dropout_rate']
        )
        
        expert_usage_logger = ExpertUsageLogger(best_model)

        logger = TensorBoardLogger("logs", name="MoE_experimental")
        csv_logger = pl.loggers.CSVLogger("logs", name="MoE_experimental")
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        checkpoint_callback = ModelCheckpoint(monitor='val_f2', mode='max') 

        trainer = pl.Trainer(
            max_epochs=300,
            logger=[logger, csv_logger],
            callbacks=[lr_monitor, checkpoint_callback, expert_usage_logger],
            accelerator='gpu',
        )
        
        train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=2048, shuffle=True)
        val_loader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=2048)
        
        trainer.fit(best_model, train_loader, val_loader)
        
        expert_usage_logger.plot_expert_usage()
        
        self.best_model, self.study = best_model, study
        self.next(self.analyze_best_model_results)

    @step
    def analyze_best_model_results(self):
        self.best_model.get_expert_activations(self.X)
        self.next(self.end)

    @step
    def end(self):
        print("Hyperparameter tuning flow completed successfully.")

if __name__ == '__main__':
    HyperparameterTuningFlow()