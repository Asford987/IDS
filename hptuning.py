import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from metaflow import FlowSpec, step, Parameter
import numpy as np
from moe import *
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset



class KerasTunerFlow(FlowSpec):

    X = Parameter('X', required=True)
    y = Parameter('y', required=True)

    @step
    def start(self):
        print(f"Data for tuning: {self.X.shape}")
        self.next(self.tuning)

    @step
    def tuning(self):
        def build_model(hp):
            model = MixtureOfExperts(self.X.shape[1], len(np.unique(self.y)), len(np.unique(self.y)), 
                                     [32,64,32], 
                                     hp.Choice('gate_hidden_units', [[16], [32], [64], [32, 16]]), 
                                     hp.Int('num_active_experts', min_value=1, max_value=5), 
                                     hp.Float('dropout_rate', min_value=0.0, max_value=0.5))
            
            model.compile(optimizer='adam', loss='sparse_categorcial_crossentropy', metrics=[
                'accuracy', 'val_accuracy',
                'AUC', 'val_AUC',
                'Precision', 'val_Precision',
                'Recall','val_Recall',
                ])
            return model
        
        tuner = BayesianOptimization(
            build_model,
            objective='val_Recall',
            max_trials=5,
            executions_per_trial=3,
            directory='logs',
            project_name='MoE_experimental')
        
        log_dir = "logs/fit/" + "MoE_experimental"
        expert_usage_logger = ExpertUsageLogger(moe_model=None)  
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        def model_build_wrapper(hp):
            model = build_model(hp)
            expert_usage_logger.moe_model = model
            return model

        tuner.hypermodel.build = model_build_wrapper

        tuner.search(self.X, 
                     self.y, 
                     epochs=300, 
                     validation_split=0.2, 
                     callbacks=[expert_usage_logger,
                                tensorboard_callback,
                                lr_scheduler])
        
        self.best_model: MixtureOfExperts = tuner.get_best_models(num_models=1)[0]
        self.best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best Hyperparameters: {self.best_hyperparameters.values}")
        self.next(self.analyze_best_model_results)

    @step
    def analyze_best_model_results(self):
        self.best_model.get_expert_activations(self.X)
        self.next(self.end)

    @step
    def end(self):
        print("KerasTuner flow completed successfully.")

# def objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim):
#     gate_hidden_units_options = {
#         "16": [16], 
#         "32": [32], 
#         "64": [64], 
#         "32_16": [32, 16]
#     }
    
#     chosen_gate_hidden_units_str = trial.suggest_categorical('gate_hidden_units', list(gate_hidden_units_options.keys()))
#     chosen_gate_hidden_units = gate_hidden_units_options[chosen_gate_hidden_units_str]

#     dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    
#     model = MixtureOfExperts(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         num_experts=output_dim,
#         expert_hidden_units=[32, 64, 32],
#         gate_hidden_units=chosen_gate_hidden_units,
#         num_active_experts=3,
#         dropout_rate=dropout_rate
#     )
    
#     expert_usage_logger = ExpertUsageLogger(model)

#     logger = TensorBoardLogger("logs", name="MoE_experimental")
#     lr_monitor = LearningRateMonitor(logging_interval='epoch')
#     checkpoint_callback = ModelCheckpoint(monitor='val_f2', mode='max') 

#     trainer = pl.Trainer(
#         max_epochs=300,
#         logger=logger,
#         callbacks=[lr_monitor, checkpoint_callback, expert_usage_logger],
#         accelerator='gpu',
#     )
    
#     train_loader = DataLoader(TensorDataset(torch.tensor(X_train.values, device='cuda'), torch.tensor(y_train, device='cuda')), batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(TensorDataset(torch.tensor(X_val.values, device='cuda'), torch.tensor(y_val, device='cuda')), batch_size=64)
    
#     trainer.fit(model, train_loader, val_loader)
    
#     val_f2 = trainer.callback_metrics["val_f2"].item()

#     return val_f2

# def tune_model(X_train, y_train, X_val, y_val, input_dim, output_dim, n_trials=20):
#     gate_hidden_units_options = {
#         "16": [16], 
#         "32": [32], 
#         "64": [64], 
#         "32_16": [32, 16]
#     }
#     study = optuna.create_study(direction="maximize")
    
#     study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim, output_dim), 
#                    n_trials=n_trials)
    
#     print(f"Best Hyperparameters: {study.best_params}")
    
#     best_params = study.best_params
#     best_gate_hidden_units = gate_hidden_units_options[best_params['gate_hidden_units']]
    
#     best_model = MixtureOfExperts(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         num_experts=output_dim, 
#         expert_hidden_units=[32, 64, 32],
#         gate_hidden_units=best_gate_hidden_units,
#         num_active_experts=3,
#         dropout_rate=best_params['dropout_rate']
#     )
    
#     expert_usage_logger = ExpertUsageLogger(best_model)

#     trainer = pl.Trainer(
#         max_epochs=50,
#         callbacks=[expert_usage_logger],
#         accelerator='gpu'
#     )
    
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
#     val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
    
#     trainer.fit(best_model, train_loader, val_loader)
    
#     expert_usage_logger.plot_expert_usage()
    
#     return best_model, study.best_params