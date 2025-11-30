"""
Main execution script for Continuous Emotion Detection.

This script orchestrates the complete pipeline:
1. Load data from .mat files
2. Preprocess and normalize features
3. Train separate models for EEG and Face features
4. Train feature-level fusion model
5. Evaluate all models and perform decision-level fusion
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
    mask_sequence,
    split_data
)
from model import network_seq
from training import train_network_seq
from evaluation import evaluate_seq, metric_seq, print_result


def main():
    """Main execution function for the emotion detection pipeline."""
    
    # Check GPU availability
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    if tf.config.list_physical_devices('GPU'):
        print("GPU devices:")
        for gpu in tf.config.list_physical_devices('GPU'):
            print(f"  - {gpu}")
    else:
        print("No GPU detected. Running on CPU.")
        print("To use GPU, ensure you have:")
        print("  1. CUDA-capable GPU")
        print("  2. tensorflow-metal (for Mac M1/M2) or tensorflow-gpu")
        print("  3. Proper CUDA/cuDNN drivers (for NVIDIA GPUs)")
    print("="*50 + "\n")
    
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
    
    # Split data into train, validation, and test sets
    print("Splitting data...")
    (train_face, val_face, test_face,
     train_eeg, val_eeg, test_eeg,
     label_train, label_val, label_test,
     mask_train,
     len_seq_train, len_seq_val, len_seq_test) = split_data(
        tensor_face_fe,
        tensor_eeg_fe,
        tensor_target,
        mask_fe,
        np.array(length_trial),
        config.TRAIN_SHARE,
        config.VAL_SHARE
    )
    
    # ========== Train EEG Network ==========
    print('\n' + '='*50)
    print('Training EEG network...')
    print('='*50)
    model_eeg = network_seq(
        tensor_eeg_fe,
        tensor_target,
        max_len,
        config.LSTM_EEG_LAYER_1,
        config.LSTM_EEG_LAYER_2,
        config.DENSE_EEG,
        dropout_rate=config.DROPOUT_RATE,
        mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER,
        loss=config.LOSS_FUNCTION
    )
    
    model_eeg = train_network_seq(
        model_eeg,
        train_eeg,
        val_eeg,
        label_train,
        label_val,
        config.EPOCHS,
        config.BATCH_SIZE,
        mask_train,
        'min',
        config.WEIGHTS_EEG
    )
    
    # ========== Train Face Network ==========
    print('\n' + '='*50)
    print('Training Face network...')
    print('='*50)
    model_face = network_seq(
        tensor_face_fe,
        tensor_target,
        max_len,
        config.LSTM_FACE_LAYER_1,
        config.LSTM_FACE_LAYER_2,
        config.DENSE_FACE,
        dropout_rate=config.DROPOUT_RATE,
        mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER,
        loss=config.LOSS_FUNCTION
    )
    
    model_face = train_network_seq(
        model_face,
        train_face,
        val_face,
        label_train,
        label_val,
        config.EPOCHS,
        config.BATCH_SIZE,
        mask_train,
        'min',
        config.WEIGHTS_FACE
    )
    
    # ========== Feature Level Fusion ==========
    print('\n' + '='*50)
    print('Training Feature Level Fusion network...')
    print('='*50)
    
    # Concatenate features
    tensor_flf = np.concatenate((tensor_face_fe, tensor_eeg_fe), axis=2)
    train_flf = np.concatenate((train_face, train_eeg), axis=2)
    val_flf = np.concatenate((val_face, val_eeg), axis=2)
    test_flf = np.concatenate((test_face, test_eeg), axis=2)
    
    model_flf = network_seq(
        tensor_flf,
        tensor_target,
        max_len,
        config.LSTM_FACE_LAYER_1 + config.LSTM_EEG_LAYER_1,
        config.LSTM_FACE_LAYER_2 + config.LSTM_EEG_LAYER_2,
        config.DENSE_FACE + config.DENSE_EEG,
        dropout_rate=config.DROPOUT_RATE,
        mask_value=config.MASK_VALUE,
        optimizer=config.OPTIMIZER,
        loss=config.LOSS_FUNCTION
    )
    
    model_flf = train_network_seq(
        model_flf,
        train_flf,
        val_flf,
        label_train,
        label_val,
        config.EPOCHS,
        config.BATCH_SIZE,
        mask_train,
        'min',
        config.WEIGHTS_FLF
    )
    
    # ========== Evaluation ==========
    print('\n' + '='*50)
    print('EVALUATION RESULTS')
    print('='*50)
    
    # Evaluate EEG model
    mse_eeg, rmse_eeg, pearson_cof_eeg, predict_test_eeg = evaluate_seq(
        model_eeg, test_eeg, label_test, config.BATCH_SIZE, len_seq_test
    )
    print_result(mse_eeg, rmse_eeg, pearson_cof_eeg, 'EEG')
    
    # Evaluate Face model
    mse_face, rmse_face, pearson_cof_face, predict_test_face = evaluate_seq(
        model_face, test_face, label_test, config.BATCH_SIZE, len_seq_test
    )
    print_result(mse_face, rmse_face, pearson_cof_face, 'Face')
    
    # Evaluate Feature Level Fusion model
    mse_flf, rmse_flf, pearson_cof_flf, predict_test_flf = evaluate_seq(
        model_flf, test_flf, label_test, config.BATCH_SIZE, len_seq_test
    )
    print_result(mse_flf, rmse_flf, pearson_cof_flf, 'Feature Level Fusion')
    
    # Decision Level Fusion
    predict_test_dlf = 0.5 * (predict_test_eeg + predict_test_face)
    mse_dlf, rmse_dlf, pearson_cof_dlf = metric_seq(
        label_test, predict_test_dlf, len_seq_test
    )
    print_result(mse_dlf, rmse_dlf, pearson_cof_dlf, 'Decision Level Fusion')
    
    print('\n' + '='*50)
    print('Training and evaluation complete!')
    print('='*50)


if __name__ == '__main__':
    main()
