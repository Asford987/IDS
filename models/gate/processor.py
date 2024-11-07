from sklearn.utils import compute_class_weight
from models.utils import reduce_mem_usage
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
        self.X = None
        self.y = None
        self.unique_labels = None
        self.label_dict = None
        self.num_classes = None

    def load_and_preprocess_data(self):
        df = pd.read_parquet(self.data_path)
        print(df['Label'].value_counts())
        df = df.drop(columns=['Timestamp'])
        df = reduce_mem_usage(df)
        df = df[df['Label'] != 'Label']
        self.df = df
        self.unique_labels = self.df['Label'].unique()
        print(f'Unique labels in dataset: {self.unique_labels}')

        # Separate Features (X)
        X = self.df.drop(columns=['Label'])
        self.X = X
        self.y = df['Label']
        
    def encode_labels(self):
        labels = self.y.unique()
        label_dict = {label: i for i, label in enumerate(labels)}
        self.label_dict = label_dict
        print(self.label_dict)
        self.y = self.y.map(self.label_dict).astype(np.uint8)
        print(self.y.value_counts())
        unique_labels_encoded = self.y.unique()
        print(f'Unique labels after encoding: {unique_labels_encoded}')
        print(f"'y' dtype: {self.y.dtype}")
        self.num_classes = len(labels)

    def split_data(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        unique_y_train_labels = np.unique(y_train)
        print(f'Unique labels in y_train: {unique_y_train_labels}')

        # Convert data to numpy arrays
        y_train = y_train.to_numpy()        
        y_test = y_test.to_numpy()        
        
        return X_train, X_test, y_train, y_test

    def compute_class_weights(self, y_train):
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        print(f'Initial class weights: {class_weight_dict}')
        return class_weight_dict

