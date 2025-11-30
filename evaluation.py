"""
Evaluation and metrics module for Continuous Emotion Detection.

This module provides functions for evaluating model predictions
and computing performance metrics.
"""

from typing import Tuple, List
import numpy as np
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential


def evaluate_seq(
    model: Sequential,
    test_data: np.ndarray,
    label_real: np.ndarray,
    batch_size: int,
    len_seq: np.ndarray
) -> Tuple[float, float, float, np.ndarray]:
    """
    Evaluate model on test sequences and compute metrics.
    
    Args:
        model: Trained Keras model
        test_data: Test input sequences (seq_num, max_len, features)
        label_real: Ground truth labels (seq_num, max_len, output_dim)
        batch_size: Batch size for prediction
        len_seq: Array of actual sequence lengths
        
    Returns:
        Tuple containing:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - pearson_cof: Mean Pearson correlation coefficient
            - label_predic: Predicted labels
    """
    # Predict test sequences
    label_predic = model.predict(test_data, batch_size=batch_size, verbose=0)
    
    # Compute metrics
    mse, rmse, pearson_cof = metric_seq(label_real, label_predic, len_seq)
    
    return mse, rmse, pearson_cof, label_predic


def metric_seq(
    label_real: np.ndarray,
    label_predic: np.ndarray,
    len_seq: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute evaluation metrics for variable-length sequences.
    
    Computes MSE, RMSE, and Pearson correlation for each sequence
    considering only the valid (non-padded) portions, then returns
    the mean across all sequences.
    
    Args:
        label_real: Ground truth labels (seq_num, max_len, output_dim)
        label_predic: Predicted labels (seq_num, max_len, output_dim)
        len_seq: Array of actual sequence lengths
        
    Returns:
        Tuple containing:
            - mean_mse: Mean squared error averaged across sequences
            - rmse: Root mean squared error
            - mean_pearson: Mean Pearson correlation coefficient
    """
    mse = []
    pearson = []
    
    for cnt in range(len(label_real)):
        # Extract valid (non-padded) portions of sequences
        predict_seq = label_predic[cnt, :int(len_seq[cnt]), :]
        real_seq = label_real[cnt, :int(len_seq[cnt]), :]
        
        # Compute metrics for this sequence
        mse.append(mean_squared_error(predict_seq, real_seq))
        # Flatten sequences for Pearson correlation (requires 1D arrays)
        pearson.append(pearsonr(predict_seq.flatten(), real_seq.flatten())[0])
    
    mean_mse = np.mean(mse)
    rmse = np.sqrt(mean_mse)
    mean_pearson = np.mean(pearson)
    
    return mean_mse, rmse, mean_pearson


def print_result(mse: float, rmse: float, pearson_cof: float, title: str) -> None:
    """
    Print evaluation results in a formatted manner.
    
    Args:
        mse: Mean squared error
        rmse: Root mean squared error
        pearson_cof: Pearson correlation coefficient
        title: Title/name of the experiment (e.g., 'EEG', 'Face', 'FLF')
    """
    print()
    print("**********", title, "**********")
    print('MSE = ', mse)
    print('RMSE = ', rmse)
    print('p = ', pearson_cof)
    print()
