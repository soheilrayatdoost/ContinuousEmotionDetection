"""
Neural network model definitions for Continuous Emotion Detection.

This module defines LSTM-based sequence models and custom loss functions
for emotion prediction from multimodal features.
"""

from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, TimeDistributed, Input
from tensorflow.keras import backend as K


def custom_loss(y_true, y_pred):
    """
    Custom loss function combining correlation and MSE.
    
    Note: This function is currently defined but not used in the main pipeline.
    The models use standard MSE loss instead.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Combined loss value (1 - mean correlation - 2*MSE)
    """
    loss = []
    dot = lambda a, b: K.batch_dot(a, b, axes=1)
    center = lambda x: x - K.mean(x)
    cov = lambda a, b: K.mean(dot(center(a), center(b)))
    correlation = lambda x: cov(x[0], x[1]) / (K.std(x[0]) * K.std(x[1]))
    
    for cnt in range(np.shape(y_pred)[2]):
        loss.append(
            correlation((y_true[cnt, :, :], y_pred[cnt, :, :])) - 
            2 * K.mean(K.square(y_true[cnt, :, :] - y_pred[cnt, :, :]))
        )
    
    return 1 - np.sum(loss) / int(np.shape(y_pred)[2])


def network_seq(
    input_tensor: np.ndarray,
    output_tensor: np.ndarray,
    max_len: int,
    lstm_size1: int,
    lstm_size2: int,
    dense_size1: int,
    dropout_rate: float = 0.4,
    mask_value: float = 0.0,
    optimizer: str = 'nadam',
    loss: str = 'mse'
) -> Sequential:
    """
    Create a sequential LSTM model for emotion prediction.
    
    The model architecture consists of:
    - Masking layer to handle variable-length sequences
    - Two LSTM layers with dropout
    - Two time-distributed dense layers with dropout
    - Linear activation for regression output
    
    Args:
        input_tensor: Input data tensor to determine feature size
        output_tensor: Output data tensor to determine output size
        max_len: Maximum sequence length
        lstm_size1: Number of units in first LSTM layer
        lstm_size2: Number of units in second LSTM layer
        dense_size1: Number of units in first dense layer (currently fixed at 32)
        dropout_rate: Dropout rate for regularization (default: 0.4)
        mask_value: Value used for masking padded sequences (default: 0.0)
        optimizer: Optimizer name (default: 'nadam')
        loss: Loss function name (default: 'mse')
        
    Returns:
        Compiled Keras Sequential model
    """
    feat_size = np.shape(input_tensor)[2]
    output_size = np.shape(output_tensor)[2]
    
    model = Sequential()
    
    # Input layer
    model.add(Input(shape=(max_len, feat_size)))
    
    # Masking layer for variable-length sequences
    model.add(Masking(mask_value=mask_value))
    
    # First LSTM layer
    model.add(LSTM(lstm_size1, return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(lstm_size2, return_sequences=True))
    model.add(Dropout(dropout_rate))
    
    # First dense layer (time-distributed)
    model.add(TimeDistributed(Dense(32)))
    model.add(Dropout(dropout_rate))
    
    # Output layer (time-distributed)
    model.add(TimeDistributed(Dense(output_size, activation='linear')))
    
    # Compile model
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
