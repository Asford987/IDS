import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.moe.moe import MixtureOfExpertsSoft, MixtureOfExpertsTopK
from models.utils import multi_f2_score


class DynamicClassWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_train, y_train, beta=2):
        super(DynamicClassWeightCallback, self).__init__()        
        self.x_train = x_train
        self.y_train = y_train
        self.beta = beta
        self.num_classes = len(np.unique(self.y_train))
    
    def on_epoch_end(self, epoch, logs=None): 
        y_pred = np.argmax(self.model.predict(self.x_train), axis=-1)
        
        # Calculate F2 score for each class
        f2_scores_train = fbeta_score(self.y_train, y_pred, beta=self.beta, average=None)

        epsilon = 1e-3
        class_weight_epoch = {}
        for cls_idx, cls in enumerate(range(self.num_classes)):
            f2_cls = f2_scores_train[cls_idx]
            class_weight_epoch[cls] = 1.0 / (f2_cls + epsilon)
        total_weight = sum(class_weight_epoch.values())
        class_weight_epoch = {k: v / total_weight * len(class_weight_epoch) for k, v in class_weight_epoch.items()}
        print(f"Updated class weights: {class_weight_epoch}")
        
        self.class_weights = class_weight_epoch


class MoETrainer:
    def __init__(self, experts_paths, gate_path, k=0, trainable_gate=True, trainable_experts=False):
        if k == 0:
            self.model = MixtureOfExpertsSoft(experts_paths, gate_path, trainable_gate=True, trainable_experts=False)
        else:
            self.model = MixtureOfExpertsTopK(experts_paths, gate_path, k=k, trainable_gate=True, trainable_experts=False)

    def train_model(self, X_train, y_train, X_val, y_val, batch_size, epochs, strategy, learning_rate=1e-3):
        # Callback assignments
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        )
        
        dynamic_weights = DynamicClassWeightCallback(
            X_train,
            y_train
        )


        # Model compilation
        with strategy.scope():
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=[multi_f2_score],
            )

        history_per_epoch = {'epoch': [], 'loss': [], 'val_loss': [], 'multi_f2_score': [], 'val_multi_f2_score': []}

        # Model fitting
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            if epoch == 0: 
                class_weight_epoch = dict(zip(compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)))
                print(f'Initial class weights: {class_weight_epoch}')
            else: class_weight_epoch = dynamic_weights.class_weights
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=1,
                batch_size=batch_size,
                verbose=1,
                class_weight=class_weight_epoch,
                callbacks=[early_stopping, reduce_lr, dynamic_weights] if epoch == 0 else [dynamic_weights],
            )

            # Updating history
            history_per_epoch['epoch'].append(epoch + 1)
            history_per_epoch['loss'].append(history.history['loss'][0])
            history_per_epoch['val_loss'].append(history.history['val_loss'][0])
            history_per_epoch['multi_f2_score'].append(history.history['multi_f2_score'][0])
            history_per_epoch['val_multi_f2_score'].append(history.history['val_multi_f2_score'][0])
            if early_stopping.stopped_epoch > 0:
                print("Early stopping triggered.")
                break
        self.history_per_epoch = history_per_epoch

        