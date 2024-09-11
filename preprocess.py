from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import logging

logging.getLogger('metaflow').setLevel(logging.WARNING)

class PreprocessingFlow(FlowSpec):
    
    outlier_threshold = Parameter('outlier_threshold', default=4)
    correlation_threshold = Parameter('correlation_threshold', default=8.0)
    train_data_path = Parameter('train_data_path', default='CIC_IoMT_2024_WiFi_MQTT_train.parquet')
    test_data_path = Parameter('test_data_path', default='CIC_IoMT_2024_WiFi_MQTT_test.parquet')
    usage_ratio = Parameter('usage_ratio', default=0.2)
    n_components = Parameter('n_components', default=15)

    @step
    def start(self):
        self.df_train = pd.read_parquet(self.train_data_path)
        def map_label(label):
            if 'Benign' in label:
                return 'benign'
            elif 'ARP_Spoofing' in label:
                return 'spoofing'
            elif 'Recon' in label:
                return 'recon'
            elif 'MQTT' in label:
                return 'MQTT'
            elif 'DoS' in label and 'DDoS' not in label:
                return 'DoS'
            elif 'DDoS' in label:
                return 'DDoS'
            
        self.df_train['label'] = self.df_train['label'].map(map_label)
        self.next(self.split_data)

    @step
    def split_data(self):
        self.df_train, _ = train_test_split(self.df_train, train_size=self.usage_ratio, stratify=self.df_train['label'])
        self.df_train, self.df_test = train_test_split(self.df_train, train_size=0.8, stratify=self.df_train['label'])

        numeric_columns = self.df_train.select_dtypes(include=[np.number]).columns
        self.df_train[numeric_columns] = self.df_train[numeric_columns].astype(np.float32)
        self.df_test[numeric_columns] = self.df_test[numeric_columns].astype(np.float32)

        self.target_train = self.df_train['label']
        self.df_train = self.df_train.drop(columns=['label', 'Drate'])
        self.target_test = self.df_test['label']
        self.df_test = self.df_test.drop(columns=['label', 'Drate'])

        self.numerical_columns = self.df_train.columns
        self.num_missing_values_df_train = self.df_train.isna().sum().sum()
        self.df_train_description = self.df_train.describe()

        print(len(np.unique(self.target_train)), len(np.unique(self.target_test)))
        self.next(self.outlier_filtering)
    
    @step
    def outlier_filtering(self):
        z_scores = np.abs(stats.zscore(self.df_train[self.numerical_columns]))
        self.outlier_mask = np.any(z_scores > self.outlier_threshold, axis=1)

        self.filtered_train = self.df_train[~self.outlier_mask]
        self.fitlered_target_train = self.target_train[~self.outlier_mask]
        print(len(np.unique(self.fitlered_target_train)))
        self.num_filtered_samples = len(self.filtered_train)        
        self.next(self.normalization)
        
    @step
    def normalization(self):
        self.scaler = StandardScaler()
        self.normalized_train = self.scaler.fit_transform(self.filtered_train)
        self.normalized_test = self.scaler.transform(self.df_test)
        self.next(self.encode_labels)
    
    @step
    def encode_labels(self):
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.target_train = self.encoder.fit_transform(self.fitlered_target_train.values.reshape(-1, 1))
        self.target_test = self.encoder.transform(self.target_test.values.reshape(-1, 1))
        
        indices = np.where(self.target_test != -1)[0]
        self.target_test = self.target_test[indices]
        self.normalized_test = self.normalized_test[indices]
        print(len(np.unique(self.target_train)), len(np.unique(self.target_test)))
        self.next(self.feature_correlation_filtering)

    @step
    def feature_correlation_filtering(self):
        self.corr_matrix = np.corrcoef(self.normalized_train, rowvar=False)
        upper_triangle_indices = np.triu_indices_from(self.corr_matrix, k=1)
        correlated_pairs = [(i, j) for i, j in zip(*upper_triangle_indices) if np.abs(self.corr_matrix[i, j]) >= self.correlation_threshold]

        self.correlated_features = set(j for _, j in correlated_pairs)
        self.pruned_train = np.delete(self.normalized_train, list(self.correlated_features), axis=1)
        self.pruned_test = np.delete(self.normalized_test, list(self.correlated_features), axis=1)
        print(f"Data after correlation filtering: {self.pruned_train.shape} (train), {self.pruned_test.shape} (test)")
        self.next(self.dimensionality_reduction)

    @step
    def dimensionality_reduction(self):
        self.name_cols = [f'PC{i}' for i in range(1, self.n_components + 1)]
        self.pca = PCA(n_components=self.n_components)
        self.reduced_train = self.pca.fit_transform(self.pruned_train)
        self.reduced_test = self.pca.transform(self.pruned_test)
        self.reduced_train = pd.DataFrame(self.reduced_train, columns=self.name_cols)
        self.reduced_test = pd.DataFrame(self.reduced_test, columns=self.name_cols)
        self.next(self.save_artifacts)
        
    @step
    def save_artifacts(self):
        import os, numpy as np
        if not os.path.exists('artifacts'): os.makedirs('artifacts')
        if not os.path.exists('artifacts/clean_data'): os.makedirs('artifacts/clean_data')
        self.reduced_train.to_csv('artifacts/clean_data/X_train.csv', index=False)
        self.reduced_test.to_csv('artifacts/clean_data/X_test.csv', index=False)
        np.save('artifacts/clean_data/y_train', self.target_train.astype(np.float32))
        np.save('artifacts/clean_data/y_test', self.target_test.astype(np.float32))
        self.next(self.end)
        
    @step
    def end(self):
        print("End of PreprocessingFlow")    
    
if __name__ == '__main__':
    PreprocessingFlow()