"""
Training utilities for Continuous Emotion Detection models.

This module provides functions for training neural network models
with checkpointing and validation.
"""

from typing import Optional
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint


def train_network_seq(
    model: Sequential,
    train_data: np.ndarray,
    val_data: np.ndarray,
    label_train: np.ndarray,
    label_val: np.ndarray,
    epochs: int,
    batch_size: int,
    mask_train: Optional[np.ndarray] = None,
    mode: str = 'min',
    path_save_net_weight: str = "weights.best.hdf5"
) -> Sequential:
    """
    Train a sequential model with validation and checkpointing.
    
    Trains the model and saves the best weights based on validation loss.
    After training, loads the best weights before returning.
    
    Args:
        model: Compiled Keras Sequential model to train
        train_data: Training input data (sequences, max_len, features)
        val_data: Validation input data (sequences, max_len, features)
        label_train: Training labels (sequences, max_len, output_dim)
        label_val: Validation labels (sequences, max_len, output_dim)
        epochs: Number of training epochs
        batch_size: Batch size for training
        mask_train: Optional sample weights for training (sequences, max_len)
        mode: Mode for checkpoint monitoring ('min' or 'max')
        path_save_net_weight: Path to save best model weights (.keras format)
        
    Returns:
        Trained model with best weights loaded
    """
    # Setup checkpoint callback
    filepath = path_save_net_weight
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode=mode
    )
    callbacks_list = [checkpoint]
    
    # Train the model
    model.fit(
        train_data,
        label_train,
        validation_data=(val_data, label_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        sample_weight=mask_train
    )
    
    # Load best weights
    model.load_weights(filepath)
    
    return model
