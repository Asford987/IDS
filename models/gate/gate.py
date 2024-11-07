from tensorflow.keras.layers import (
    Add,
    Activation,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MultiHeadAttention,
    Reshape,
    GlobalAveragePooling1D,
    LayerNormalization,
)
from tensorflow.keras.models import Model


class ModelBuilder:
    """Builds different types of models."""
    def __init__(self, input_shape, num_classes, strategy):
        self.input_shape = input_shape
        self.num_classes = num_classes
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
            x = Dense(128, activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = residual_block(x, 128)
            x = residual_block(x, 128)
            x = residual_block(x, 128)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model

    # def build_cnn_model(self):
    #     def residual_block_cnn(x, filters, kernel_size=3, stride=1, dropout_rate=0.0):
    #         # First Convolutional Layer
    #         shortcut = x
    #         x = Conv1D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    #         x = BatchNormalization()(x)
    #         x = Activation('relu')(x)
            
    #         # Second Convolutional Layer
    #         x = Conv1D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    #         x = BatchNormalization()(x)
            
    #         # Residual Connection
    #         x = Add()([shortcut, x])
    #         x = Activation('relu')(x)
            
    #         # Optional Pooling and Dropout
    #         if stride > 1:
    #             x = MaxPooling1D(pool_size=2)(x)
    #         if dropout_rate > 0:
    #             x = Dropout(dropout_rate)(x)
            
    #         return x
    #     with self.strategy.scope():
    #         inputs = Input(shape=(self.input_shape,))
    #         x = Reshape((self.input_shape, 1))(inputs)
            
    #         # First Residual Block
    #         x = residual_block_cnn(x, filters=64, kernel_size=3, stride=1, dropout_rate=0.2)
            
    #         # # Second Residual Block
    #         # x = residual_block_cnn(x, filters=64, kernel_size=3, stride=1, dropout_rate=0.4)
            
    #         # # Third Residual Block
    #         # x = residual_block_cnn(x, filters=64, kernel_size=3, stride=1, dropout_rate=0.5)
            
    #         # Global Average Pooling
    #         x = GlobalAveragePooling1D()(x)
            
    #         # Fully Connected Layer
    #         x = Dense(64, activation='relu')(x)
    #         x = BatchNormalization()(x)
    #         x = Dropout(0.5)(x)
            
    #         # Output Layer
    #         outputs = Dense(self.num_classes, activation='softmax')(x)
            
    #         model = Model(inputs=inputs, outputs=outputs)
    #         return model
    def build_cnn_model(self):
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            x = Conv1D(64, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Conv1D(64, kernel_size=3, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.2)(x)
            x = Flatten()(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            model = Model(inputs=inputs, outputs=outputs)
            return model


    def build_lstm_model(self):
        def lstm_block(x, units, dropout_rate=0.0, return_sequences=True):
                x = LSTM(units, return_sequences=return_sequences)(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)
                return x
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            
            # # First LSTM Block
            # x = lstm_block(x, units=64, dropout_rate=0.2, return_sequences=True)
            
            # # Second LSTM Block
            # x = lstm_block(x, units=64, dropout_rate=0.2, return_sequences=True)
            
            # Third LSTM Block
            x = lstm_block(x, units=64, dropout_rate=0.2, return_sequences=False)
            
            # Fully Connected Layer
            x = Dense(64, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            # Output Layer
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model

    
    
    def build_attention_model(self, num_heads=4, ff_dim=64):
        def attention_block(x, num_heads, key_dim, ff_dim, rate=0.1):
            # Multi-Head Attention
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
            attn_output = Dropout(rate)(attn_output)
            # Residual Connection and Layer Normalization
            out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed-Forward Network
            ffn_output = Dense(ff_dim, activation='relu')(out1)
            ffn_output = Dense(x.shape[-1])(ffn_output)
            ffn_output = Dropout(rate)(ffn_output)
            # Residual Connection and Layer Normalization
            out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
            return out2
        with self.strategy.scope():
            inputs = Input(shape=(self.input_shape,))
            x = Reshape((self.input_shape, 1))(inputs)
            
            # Attention Blocks
            # x = attention_block(x, num_heads=num_heads, key_dim=32, ff_dim=ff_dim, rate=0.1)
            x = attention_block(x, num_heads=num_heads, key_dim=32, ff_dim=ff_dim, rate=0.2)
            
            # Global Average Pooling
            x = GlobalAveragePooling1D()(x)
            x = Dropout(0.2)(x)
            
            # Fully Connected Layer
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.2)(x)
            
            # Output Layer
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model
