from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.models import Model


class ModelBuilder:
    """Builds different types of models."""
    def __init__(self, input_shape, strategy):
        self.input_shape = input_shape
        self.strategy = strategy

    def build_mlp_residual_model(self):
        with self.strategy.scope():
            def residual_block(x, units, dropout_rate=0.2):
                shortcut = x
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)
                x = Dense(units, activation='relu')(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)
                x = Add()([shortcut, x])
                return x

            inputs = Input(shape=(self.input_shape,))
            x = Dense(64, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model

    def build_cnn_model(self):
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            x = Conv1D(32, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Conv1D(32, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model

    def build_lstm_model(self):
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            x = LSTM(32, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model

    def build_attention_model(self, num_heads=4):
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=32)(x, x)
            x = Flatten()(attention_output)
            x = Dense(32, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model