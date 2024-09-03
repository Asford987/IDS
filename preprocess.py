from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

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
        self.df_test = pd.read_parquet(self.test_data_path)
        
        df_combined = pd.concat([self.df_train, self.df_test])
        df_sampled, _ = train_test_split(df_combined, train_size=self.usage_ratio, stratify=df_combined['label'])
        self.df_train: pd.DataFrame = df_sampled[df_sampled.index.isin(self.df_train.index)]
        self.df_test: pd.DataFrame = df_sampled[df_sampled.index.isin(self.df_test.index)]
        numeric_columns = self.df_train.select_dtypes(include=[np.number]).columns
        self.df_train[numeric_columns] = self.df_train[numeric_columns].astype(np.float32)
        self.df_test[numeric_columns] = self.df_test[numeric_columns].astype(np.float32)
        
        buffer = StringIO()
        self.df_train.info(buf=buffer)
        self.train_data_info = buffer.getvalue()
        self.train_data_unique = self.df_train.nunique().to_dict()
        
        self.numerical_columns = self.df_train.select_dtypes(exclude=['category']).columns
        self.target_train = self.df_train['label']
        self.df_train = self.df_train.drop(columns=['label', 'Drate'])
        self.target_test = self.df_test['label']
        self.df_test = self.df_test.drop(columns=['label', 'Drate'])
        self.num_missing_values_df_train = self.df_train.isna().sum().sum()
        self.df_train_description = self.df_train.describe()
        self.next(self.outlier_filtering)

    @step
    def outlier_filtering(self):
        z_scores = np.abs(stats.zscore(self.df_train[self.numerical_columns]))
        self.outlier_mask = np.any(z_scores > self.outlier_threshold, axis=1)

        self.filtered_train = self.df_train[~self.outlier_mask]
        self.fitlered_target_train = self.target_train[~self.outlier_mask]
        
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
        self.target_train = self.encoder.fit_transform(self.target_train.values.reshape(-1, 1))
        self.target_test = self.encoder.transform(self.target_test.values.reshape(-1, 1))
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
        self.next(self.end)
        
    @step
    def end(self):
        print("End of PreprocessingFlow")    
    