"""
Data preprocessing module for Continuous Emotion Detection.

This module contains functions for normalizing, reshaping, and splitting
the emotion detection dataset.
"""

from typing import Dict, List, Tuple
import numpy as np
from sklearn import preprocessing


def normalize_data_subject(
    annotated_trial: np.ndarray,
    features_label: Dict[int, Dict],
    data_num: int,
    annotation: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Normalize facial and EEG features per subject.
    
    Extracts face features, EEG features, and targets from loaded data,
    then normalizes them on a per-subject basis (standardization to zero
    mean and unit variance).
    
    Args:
        annotated_trial: Array of [participant_id, trial_id] pairs
        features_label: Dictionary mapping trial index to loaded .mat contents
        data_num: Total number of samples across all trials
        annotation: Dictionary containing 'trials_included' array
        
    Returns:
        Tuple containing:
            - face_fe: Normalized facial features (data_num, 38)
            - eeg_fe: Normalized EEG features (data_num, 128)
            - target: Target emotion values (data_num, 1)
            - length_trial: List of sequence lengths for each trial
    """
    face_fe = np.zeros((data_num, 38))
    eeg_fe = np.zeros((data_num, 128))
    target = np.zeros((data_num, 1))
    start_trial = 0
    trial_num = int(np.shape(annotated_trial)[0])
    length_trial = []
    part_num_sample = 0
    part_trial = []
    
    for cnt, par_trial in enumerate(annotated_trial):
        # Get trial length and compute indices
        length_trial.append(int(np.shape(features_label[cnt]['face_feats'])[0]))
        end_trial = start_trial + int(length_trial[cnt])
        
        # Extract features and targets
        face_fe[start_trial:end_trial, :] = features_label[cnt]['face_feats'][:, 0:38]
        eeg_fe[start_trial:end_trial, :] = features_label[cnt]['eeg_band_feats_full']
        target[start_trial:end_trial, :] = np.transpose(features_label[cnt]['target'])
        
        start_trial = end_trial
        
        # Normalize per subject
        if cnt > 0:
            if (annotation['trials_included'][cnt, 0] == 
                annotation['trials_included'][cnt - 1, 0] and 
                cnt != trial_num - 1):
                part_num_sample += int(length_trial[cnt])
            else:
                if cnt == trial_num - 1:
                    part_num_sample += int(length_trial[cnt])
                start = int(np.sum(part_trial))
                face_fe[start:int(start + part_num_sample), :] = preprocessing.scale(
                    face_fe[start:int(start + part_num_sample), :]
                )
                eeg_fe[start:int(start + part_num_sample), :] = preprocessing.scale(
                    eeg_fe[start:int(start + part_num_sample), :]
                )
                part_trial.append(part_num_sample)
                part_num_sample = int(length_trial[cnt])
        else:
            part_num_sample = int(length_trial[cnt])
    
    return face_fe, eeg_fe, target, length_trial


def array_to_sequence(
    np_array: np.ndarray,
    max_len: int,
    seq_num: int,
    length_trial: List[int]
) -> np.ndarray:
    """
    Convert flat array to 3D sequence tensor with padding.
    
    Transforms a 2D array into a 3D tensor where each sequence is padded
    to the maximum sequence length.
    
    Args:
        np_array: Input 2D array (total_samples, features)
        max_len: Maximum sequence length for padding
        seq_num: Number of sequences
        length_trial: List of actual lengths for each sequence
        
    Returns:
        3D tensor of shape (seq_num, max_len, features) with zero-padding
    """
    tensor_seq = np.zeros((seq_num, max_len, int(np.shape(np_array)[1])))
    
    start_trial = 0
    for cnt, len_trial in enumerate(length_trial):
        end_trial = start_trial + int(len_trial)
        tensor_seq[cnt, :int(len_trial), :] = np_array[start_trial:end_trial, :]
        start_trial = end_trial
    
    return tensor_seq


def mask_sequence(max_len: int, seq_num: int, length_trial: List[int]) -> np.ndarray:
    """
    Create binary mask for variable-length sequences.
    
    Generates a mask array where 1 indicates valid data and 0 indicates padding.
    
    Args:
        max_len: Maximum sequence length
        seq_num: Number of sequences
        length_trial: List of actual lengths for each sequence
        
    Returns:
        Binary mask array of shape (seq_num, max_len)
    """
    mask_seq = np.zeros((seq_num, max_len))
    
    for cnt, len_trial in enumerate(length_trial):
        mask_seq[cnt, :int(len_trial)] = np.ones((1, int(len_trial)))
    
    return mask_seq


def split_data(
    tensor_face_fe: np.ndarray,
    tensor_eeg_fe: np.ndarray,
    tensor_target: np.ndarray,
    mask_fe: np.ndarray,
    length_trial: np.ndarray,
    train_share: float,
    val_share: float
) -> Tuple[np.ndarray, ...]:
    """
    Split dataset into train, validation, and test sets with shuffling.
    
    Args:
        tensor_face_fe: Face features tensor (seq_num, max_len, face_features)
        tensor_eeg_fe: EEG features tensor (seq_num, max_len, eeg_features)
        tensor_target: Target tensor (seq_num, max_len, 1)
        mask_fe: Mask tensor (seq_num, max_len)
        length_trial: Array of sequence lengths
        train_share: Proportion of data for training (e.g., 0.6)
        val_share: Proportion of data for validation (e.g., 0.3)
        
    Returns:
        Tuple containing 13 elements:
            - train_face, val_face, test_face: Face feature splits
            - train_eeg, val_eeg, test_eeg: EEG feature splits
            - label_train, label_val, label_test: Target splits
            - mask_train: Training mask
            - len_seq_train, len_seq_val, len_seq_test: Sequence length splits
    """
    # Calculate split sizes
    train_size = int(len(tensor_face_fe) * train_share)
    val_size = int(len(tensor_face_fe) * val_share)
    
    # Shuffle indices
    indices = np.arange(len(tensor_face_fe), dtype='int')
    np.random.shuffle(indices)
    
    # Apply shuffling to all arrays
    data_face = tensor_face_fe[indices, :, :]
    data_eeg = tensor_eeg_fe[indices, :, :]
    labels = tensor_target[indices, :, :]
    data_mask = mask_fe[indices, :]
    len_tr_rnd = length_trial[indices]
    
    # Split face features
    train_face = data_face[0:train_size, :, :]
    val_face = data_face[train_size:train_size + val_size, :, :]
    test_face = data_face[train_size + val_size:, :, :]
    
    # Split EEG features
    train_eeg = data_eeg[0:train_size, :, :]
    val_eeg = data_eeg[train_size:train_size + val_size, :, :]
    test_eeg = data_eeg[train_size + val_size:, :, :]
    
    # Split sequence lengths
    len_seq_train = len_tr_rnd[0:train_size]
    len_seq_val = len_tr_rnd[train_size:train_size + val_size]
    len_seq_test = len_tr_rnd[train_size + val_size:]
    
    # Split labels
    label_train = labels[0:train_size, :, :]
    label_val = labels[train_size:train_size + val_size, :, :]
    label_test = labels[train_size + val_size:, :, :]
    
    # Training mask only
    mask_train = data_mask[0:train_size, :]
    
    return (train_face, val_face, test_face,
            train_eeg, val_eeg, test_eeg,
            label_train, label_val, label_test,
            mask_train,
            len_seq_train, len_seq_val, len_seq_test)
