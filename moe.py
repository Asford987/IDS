import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import fbeta_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

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
    def __init__(self, input_dim, output_dim, num_experts, expert_hidden_units, gate_hidden_units, num_active_experts, dropout_rate, learning_rate=1e-3):
        super(MixtureOfExperts, self).__init__()
        self.save_hyperparameters()

        self.experts = nn.ModuleList([ExpertModel(input_dim, output_dim, expert_hidden_units, dropout_rate) for _ in range(num_experts)])
        self.gate = GateModel(input_dim, num_experts, gate_hidden_units, dropout_rate)
        self.num_active_experts = num_active_experts
        self.expert_usage_count = torch.zeros(num_experts, dtype=torch.float32)

        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        gate_output = self.gate(x)

        expert_usage_count_adjusted = self.expert_usage_count + 1e-10
        importance_scores = gate_output / expert_usage_count_adjusted

        top_n_expert_indices = torch.argsort(importance_scores, dim=1, descending=True)[:, :self.num_active_experts]
        selected_expert_indices = top_n_expert_indices.view(-1)

        self.expert_usage_count += torch.bincount(selected_expert_indices, minlength=len(self.experts)).float()

        mask = torch.sum(F.one_hot(top_n_expert_indices, num_classes=len(self.experts)), dim=1)
        masked_gate_output = gate_output * mask
        normalized_gate_output = masked_gate_output / (torch.sum(masked_gate_output, dim=1, keepdim=True) + 1e-7)

        masked_expert_outputs = torch.stack([expert_outputs[:, i] * normalized_gate_output[:, i].unsqueeze(1)
                                              for i in range(len(self.experts))], dim=1)
        final_output = torch.sum(masked_expert_outputs, dim=1)

        return final_output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        f2_score = fbeta_score(y.cpu().numpy(), preds.cpu().numpy(), beta=2, average='macro')
        self.log('val_f2', f2_score, prog_bar=True, sync_dist=True)
        return {'val_f2': f2_score}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=5, verbose=True
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

    def on_fit_start(self):
        self.expert_usage_count = self.expert_usage_count.to(self.device)

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
