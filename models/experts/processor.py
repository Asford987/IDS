from sklearn.utils import compute_class_weight
from models.utils import reduce_mem_usage, multi_f2_score
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Handles data loading and preprocessing."""
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_scaled = None
        self.y = None
        self.unique_labels = None

    def load_and_preprocess_data(self):
        df = pd.read_parquet(self.data_path)
        print(df['Label'].value_counts())
        df = df.drop(columns=['Timestamp'])
        df = reduce_mem_usage(df)
        df = df[df['Label'] != 'Label']
        self.df = df
        self.unique_labels = df['Label'].unique()
        print(f'Unique labels in dataset: {self.unique_labels}')

        # Separate Features (X)
        X = df.drop(columns=['Label'])
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(X)

    def binarize_labels(self, binarize_on_label):
        y = (self.df['Label'] == binarize_on_label).astype(np.uint8)
        print(y.value_counts())
        unique_binary_labels = y.unique()
        print(f'Unique labels after binarization: {unique_binary_labels}')
        print(f"'y' dtype: {y.dtype}")
        self.y = y

    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        unique_y_train_labels = np.unique(y_train)
        print(f'Unique labels in y_train: {unique_y_train_labels}')
        return X_train, X_test, y_train, y_test

    def compute_class_weights(self, y_train):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train.ravel()
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f'Initial class weights: {class_weight_dict}')
        return class_weight_dict
