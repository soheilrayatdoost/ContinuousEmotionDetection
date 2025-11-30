"""
Data loading module for Continuous Emotion Detection.

This module handles loading of .mat files containing facial features,
EEG features, and emotion labels.
"""

from typing import Dict, Tuple
import numpy as np
import scipy.io as sio


def load_data(data_folder: str, annotated_trial: np.ndarray) -> Tuple[Dict[int, Dict], int]:
    """
    Load feature data from .mat files for all annotated trials.
    
    Args:
        data_folder: Path to the folder containing feature .mat files
        annotated_trial: Array of [participant_id, trial_id] pairs to load
        
    Returns:
        Tuple containing:
            - Dictionary mapping trial index to loaded .mat file contents
            - Total number of data samples across all trials
            
    Raises:
        FileNotFoundError: If a specified .mat file cannot be found
        IOError: If there's an error reading a .mat file
    """
    features_label = {}
    data_num = 0
    
    for cnt, par_trial in enumerate(annotated_trial):
        data_address = "{}P{}-features-resamp4hz-trial-{}.mat".format(
            data_folder, par_trial[0], par_trial[1]
        )
        
        try:
            features_label[cnt] = sio.loadmat(data_address)
            data_num += np.shape(features_label[cnt]['face_feats'])[0]
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file: {data_address}")
        except Exception as e:
            raise IOError(f"Error loading {data_address}: {str(e)}")
    
    return features_label, data_num
