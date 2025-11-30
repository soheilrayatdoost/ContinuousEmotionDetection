"""
Configuration module for Continuous Emotion Detection.

This module contains all hyperparameters, file paths, and configuration
constants used throughout the emotion detection pipeline.
"""

# Data parameters
TRAIN_SHARE = 0.6
VAL_SHARE = 0.3
TEST_SHARE = 0.1  # Derived: 1 - TRAIN_SHARE - VAL_SHARE

# EEG Network parameters
LSTM_EEG_LAYER_1 = 64
LSTM_EEG_LAYER_2 = 32
DENSE_EEG = 32

# Face Network parameters
LSTM_FACE_LAYER_1 = 19
LSTM_FACE_LAYER_2 = 10
DENSE_FACE = 10

# Training parameters
EPOCHS = 100
BATCH_SIZE = 20
DROPOUT_RATE = 0.4

# File paths
DATA_FOLDER = './data/Features/'
ANNOTATION_PATH = './data/lable_continous_Mahnob.mat'
WEIGHTS_EEG = 'weights.bestEEG.keras'
WEIGHTS_FACE = 'weights.bestFace.keras'
WEIGHTS_FLF = 'weights.bestFLF.keras'

# Model parameters
OPTIMIZER = 'nadam'
LOSS_FUNCTION = 'mse'
MASK_VALUE = 0.0

# Feature dimensions
FACE_FEATURE_DIM = 38
EEG_FEATURE_DIM = 128
TARGET_DIM = 1
