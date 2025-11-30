"""
Cross-validation utilities for Continuous Emotion Detection.

This module provides functions for performing k-fold cross-validation
on emotion detection models.
"""

from typing import List, Tuple, Dict
import numpy as np
from tensorflow.keras.models import Sequential


def create_k_folds(
    num_samples: int,
    k: int = 10,
    shuffle: bool = True,
    random_seed: int = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create k-fold split indices for cross-validation.
    
    Args:
        num_samples: Total number of samples
        k: Number of folds (default: 10)
        shuffle: Whether to shuffle indices before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    indices = np.arange(num_samples)
    
    if shuffle:
        if random_seed is not None:
            np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    fold_sizes = np.full(k, num_samples // k, dtype=int)
    fold_sizes[:num_samples % k] += 1
    
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices, test_indices))
        current = stop
    
    return folds


def split_data_k_fold(
    tensor_face_fe: np.ndarray,
    tensor_eeg_fe: np.ndarray,
    tensor_target: np.ndarray,
    mask_fe: np.ndarray,
    length_trial: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    val_split: float = 0.2
) -> Tuple[np.ndarray, ...]:
    """
    Split data for one fold of cross-validation.
    
    Args:
        tensor_face_fe: Face features tensor
        tensor_eeg_fe: EEG features tensor
        tensor_target: Target tensor
        mask_fe: Mask tensor
        length_trial: Array of sequence lengths
        train_indices: Indices for training set
        test_indices: Indices for test set
        val_split: Proportion of training data to use for validation
        
    Returns:
        Tuple containing train, validation, and test splits for all data
    """
    # Extract test set
    test_face = tensor_face_fe[test_indices, :, :]
    test_eeg = tensor_eeg_fe[test_indices, :, :]
    test_target = tensor_target[test_indices, :, :]
    len_seq_test = length_trial[test_indices]
    
    # Split training set into train and validation
    num_train = len(train_indices)
    num_val = int(num_train * val_split)
    
    # Shuffle train indices for validation split
    train_indices_shuffled = train_indices.copy()
    np.random.shuffle(train_indices_shuffled)
    
    val_indices = train_indices_shuffled[:num_val]
    train_indices_final = train_indices_shuffled[num_val:]
    
    # Extract training set
    train_face = tensor_face_fe[train_indices_final, :, :]
    train_eeg = tensor_eeg_fe[train_indices_final, :, :]
    train_target = tensor_target[train_indices_final, :, :]
    mask_train = mask_fe[train_indices_final, :]
    len_seq_train = length_trial[train_indices_final]
    
    # Extract validation set
    val_face = tensor_face_fe[val_indices, :, :]
    val_eeg = tensor_eeg_fe[val_indices, :, :]
    val_target = tensor_target[val_indices, :, :]
    len_seq_val = length_trial[val_indices]
    
    return (train_face, val_face, test_face,
            train_eeg, val_eeg, test_eeg,
            train_target, val_target, test_target,
            mask_train,
            len_seq_train, len_seq_val, len_seq_test)


def aggregate_fold_results(fold_results: List[Dict[str, Tuple[float, float, float]]]) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """
    Aggregate results across all folds.
    
    Args:
        fold_results: List of dictionaries containing results for each fold
        
    Returns:
        Dictionary with mean and std for each metric and model
        Format: {model: (mse_mean, mse_std, rmse_mean, rmse_std, pearson_mean, pearson_std)}
    """
    models = fold_results[0].keys()
    aggregated = {}
    
    for model in models:
        mse_values = [fold[model][0] for fold in fold_results]
        rmse_values = [fold[model][1] for fold in fold_results]
        pearson_values = [fold[model][2] for fold in fold_results]
        
        aggregated[model] = (
            np.mean(mse_values),
            np.std(mse_values),
            np.mean(rmse_values),
            np.std(rmse_values),
            np.mean(pearson_values),
            np.std(pearson_values)
        )
    
    return aggregated


def print_fold_results(fold_num: int, results: Dict[str, Tuple[float, float, float]]) -> None:
    """
    Print results for a single fold.
    
    Args:
        fold_num: Fold number (1-indexed)
        results: Dictionary containing results for each model
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num} RESULTS")
    print('='*70)
    
    for model, (mse, rmse, pearson) in results.items():
        print(f"\n{model}:")
        print(f"  MSE   = {mse:.6f}")
        print(f"  RMSE  = {rmse:.6f}")
        print(f"  Pearson = {pearson:.6f}")


def print_aggregated_results(aggregated: Dict[str, Tuple[float, float, float, float, float, float]]) -> None:
    """
    Print aggregated results across all folds.
    
    Args:
        aggregated: Dictionary with mean and std for each metric and model
    """
    print(f"\n{'='*70}")
    print("10-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)")
    print('='*70)
    
    for model, (mse_mean, mse_std, rmse_mean, rmse_std, pearson_mean, pearson_std) in aggregated.items():
        print(f"\n{model}:")
        print(f"  MSE     = {mse_mean:.6f} ± {mse_std:.6f}")
        print(f"  RMSE    = {rmse_mean:.6f} ± {rmse_std:.6f}")
        print(f"  Pearson = {pearson_mean:.6f} ± {pearson_std:.6f}")
    
    print(f"\n{'='*70}")
