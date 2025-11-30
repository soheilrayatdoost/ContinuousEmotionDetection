"""
Main execution script with 10-fold cross-validation for Continuous Emotion Detection.

This script performs 10-fold cross-validation:
1. Load data from .mat files
2. Preprocess and normalize features
3. For each fold:
   - Train separate models for EEG and Face features
   - Train feature-level fusion model
   - Evaluate all models and perform decision-level fusion
4. Report aggregated results across all folds
"""

import numpy as np
import scipy.io as sio
import tensorflow as tf

# Import configuration
import config

# Import custom modules
from data_loader import load_data
from preprocessing import (
    normalize_data_subject,
    array_to_sequence,
    mask_sequence
)
from cross_validation import (
    create_k_folds,
    split_data_k_fold,
    aggregate_fold_results,
    print_fold_results,
    print_aggregated_results
)
from model import network_seq
from training import train_network_seq
from evaluation import evaluate_seq, metric_seq


def train_and_evaluate_fold(
    fold_num: int,
    train_face: np.ndarray,
    val_face: np.ndarray,
    test_face: np.ndarray,
    train_eeg: np.ndarray,
    val_eeg: np.ndarray,
    test_eeg: np.ndarray,
    label_train: np.ndarray,
    label_val: np.ndarray,
    label_test: np.ndarray,
    mask_train: np.ndarray,
    len_seq_test: np.ndarray,
    max_len: int,
    tensor_face_fe: np.ndarray,
    tensor_eeg_fe: np.ndarray,
    tensor_target: np.ndarray
) -> dict:
    """
    Train and evaluate all models for one fold.
    
    Returns:
        Dictionary with results for each model
    """
    print(f"\n{'='*70}")
    print(f"FOLD {fold_num}/10")
    print('='*70)
    
    # Train EEG model
    print(f"\nTraining EEG model (Fold {fold_num})...")
    model_eeg = network_seq(
        tensor_eeg_fe, tensor_target, max_len,
        config.LSTM_EEG_LAYER_1, config.LSTM_EEG_LAYER_2, config.DENSE_EEG,
        dropout_rate=config.DROPOUT_RATE, mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER, loss=config.LOSS_FUNCTION
    )
    model_eeg = train_network_seq(
        model_eeg, train_eeg, val_eeg, label_train, label_val,
        config.EPOCHS, config.BATCH_SIZE, mask_train, 'min',
        f'weights_fold{fold_num}_EEG.keras'
    )
    
    # Train Face model
    print(f"\nTraining Face model (Fold {fold_num})...")
    model_face = network_seq(
        tensor_face_fe, tensor_target, max_len,
        config.LSTM_FACE_LAYER_1, config.LSTM_FACE_LAYER_2, config.DENSE_FACE,
        dropout_rate=config.DROPOUT_RATE, mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER, loss=config.LOSS_FUNCTION
    )
    model_face = train_network_seq(
        model_face, train_face, val_face, label_train, label_val,
        config.EPOCHS, config.BATCH_SIZE, mask_train, 'min',
        f'weights_fold{fold_num}_Face.keras'
    )
    
    # Feature Level Fusion
    print(f"\nTraining Feature Level Fusion model (Fold {fold_num})...")
    train_flf = np.concatenate((train_face, train_eeg), axis=2)
    val_flf = np.concatenate((val_face, val_eeg), axis=2)
    test_flf = np.concatenate((test_face, test_eeg), axis=2)
    
    tensor_flf = np.concatenate((tensor_face_fe, tensor_eeg_fe), axis=2)
    model_flf = network_seq(
        tensor_flf, tensor_target, max_len,
        config.LSTM_FACE_LAYER_1 + config.LSTM_EEG_LAYER_1,
        config.LSTM_FACE_LAYER_2 + config.LSTM_EEG_LAYER_2,
        config.DENSE_FACE + config.DENSE_EEG,
        dropout_rate=config.DROPOUT_RATE, mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER, loss=config.LOSS_FUNCTION
    )
    model_flf = train_network_seq(
        model_flf, train_flf, val_flf, label_train, label_val,
        config.EPOCHS, config.BATCH_SIZE, mask_train, 'min',
        f'weights_fold{fold_num}_FLF.keras'
    )
    
    # Evaluate all models
    print(f"\nEvaluating Fold {fold_num}...")
    mse_eeg, rmse_eeg, pearson_eeg, predict_test_eeg = evaluate_seq(
        model_eeg, test_eeg, label_test, config.BATCH_SIZE, len_seq_test
    )
    
    mse_face, rmse_face, pearson_face, predict_test_face = evaluate_seq(
        model_face, test_face, label_test, config.BATCH_SIZE, len_seq_test
    )
    
    mse_flf, rmse_flf, pearson_flf, predict_test_flf = evaluate_seq(
        model_flf, test_flf, label_test, config.BATCH_SIZE, len_seq_test
    )
    
    # Decision Level Fusion
    predict_test_dlf = 0.5 * (predict_test_eeg + predict_test_face)
    mse_dlf, rmse_dlf, pearson_dlf = metric_seq(
        label_test, predict_test_dlf, len_seq_test
    )
    
    # Store results
    results = {
        'EEG': (mse_eeg, rmse_eeg, pearson_eeg),
        'Face': (mse_face, rmse_face, pearson_face),
        'Feature Level Fusion': (mse_flf, rmse_flf, pearson_flf),
        'Decision Level Fusion': (mse_dlf, rmse_dlf, pearson_dlf)
    }
    
    return results


def main():
    """Main execution function with 10-fold cross-validation."""
    
    # Check GPU availability
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    if tf.config.list_physical_devices('GPU'):
        print("GPU devices:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
    else:
        print("No GPU detected. Running on CPU.")
    print("="*70 + "\n")
    
    # Load annotation file
    print("Loading annotations...")
    annotation = sio.loadmat(config.ANNOTATION_PATH)
    
    # Load feature data
    print("Loading feature data...")
    features_label, data_num = load_data(
        config.DATA_FOLDER,
        annotation['trials_included']
    )
    
    # Normalize data per subject
    print("Normalizing data...")
    face_fe, eeg_fe, target, length_trial = normalize_data_subject(
        annotation['trials_included'],
        features_label,
        data_num,
        annotation
    )
    
    # Prepare sequence tensors
    max_len = int(np.max(length_trial))
    seq_num = len(annotation['trials_included'])
    
    print("Converting to sequences...")
    tensor_face_fe = array_to_sequence(face_fe, max_len, seq_num, length_trial)
    tensor_eeg_fe = array_to_sequence(eeg_fe, max_len, seq_num, length_trial)
    tensor_target = array_to_sequence(target, max_len, seq_num, length_trial)
    mask_fe = mask_sequence(max_len, seq_num, length_trial)
    
    # Create 10-fold splits
    print("\nCreating 10-fold cross-validation splits...")
    k_folds = create_k_folds(seq_num, k=10, shuffle=True, random_seed=42)
    
    # Store results for all folds
    all_fold_results = []
    
    # Perform cross-validation
    for fold_num, (train_indices, test_indices) in enumerate(k_folds, 1):
        # Split data for this fold
        (train_face, val_face, test_face,
         train_eeg, val_eeg, test_eeg,
         label_train, label_val, label_test,
         mask_train,
         len_seq_train, len_seq_val, len_seq_test) = split_data_k_fold(
            tensor_face_fe, tensor_eeg_fe, tensor_target, mask_fe,
            np.array(length_trial), train_indices, test_indices, val_split=0.2
        )
        
        # Train and evaluate for this fold
        fold_results = train_and_evaluate_fold(
            fold_num,
            train_face, val_face, test_face,
            train_eeg, val_eeg, test_eeg,
            label_train, label_val, label_test,
            mask_train, len_seq_test, max_len,
            tensor_face_fe, tensor_eeg_fe, tensor_target
        )
        
        # Print fold results
        print_fold_results(fold_num, fold_results)
        
        # Store results
        all_fold_results.append(fold_results)
    
    # Aggregate and print final results
    print("\n" + "="*70)
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print("="*70)
    
    aggregated_results = aggregate_fold_results(all_fold_results)
    print_aggregated_results(aggregated_results)
    
    print("\n" + "="*70)
    print("10-FOLD CROSS-VALIDATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
