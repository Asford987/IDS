import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K # type: ignore
import pandas as pd

def reduce_mem_usage(df):
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Initial memory usage: {start_mem:.2f} MB')
    for col in df.columns:
        col_type = df[col].dtype
        if col_type.kind in ['i', 'u', 'f']:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.api.types.is_integer_dtype(col_type):
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            if df[col].dtype == 'object' and df[col].nunique() / len(df[col]) < 0.5:
                df[col] = df[col].astype('category')
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Optimized memory usage: {end_mem:.2f} MB')
    print(f'Reduced by {(start_mem - end_mem) / start_mem * 100:.1f}%')
    return df


def multi_f2_score(y_true, y_pred):
    beta = 2
    # Convert y_true to one-hot encoding
    y_true = tf.cast(y_true, 'int32')
    num_classes = tf.shape(y_pred)[-1]
    y_true = tf.one_hot(y_true, depth=num_classes)

    # Convert y_pred to binary predictions
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.one_hot(y_pred, depth=num_classes)

    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')

    # Calculate true positives, false positives, false negatives
    tp = K.sum(y_true * y_pred, axis=0)
    fp = K.sum((1 - y_true) * y_pred, axis=0)
    fn = K.sum(y_true * (1 - y_pred), axis=0)

    # Calculate precision and recall
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    beta_squared = beta ** 2
    f2 = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())

    # Compute mean F2 across all classes
    f2 = K.mean(f2)
    return f2

def f2_score(y_true, y_pred):
    beta = 2
    y_pred = K.round(y_pred)
    # Cast y_true and y_pred to float32
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    beta_squared = beta ** 2
    f2 = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
    return f2