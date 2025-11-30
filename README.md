# Continuous Emotion Detection

Analysis of EEG Signals and Facial Expressions for Continuous Emotion Detection using LSTM neural networks.

This codebase implements multimodal emotion detection combining EEG signals and facial expressions, with support for feature-level and decision-level fusion strategies.

> **Note**: This repository is an implementation based on the methodology described in the paper. It is not the original code used to produce the results in the published paper. This is a modernized, refactored version with updated dependencies and best practices.

## 🚀 Features

- **Multimodal Learning**: Combines EEG and facial expression features
- **LSTM-based Architecture**: Handles variable-length time series data
- **Multiple Fusion Strategies**:
  - EEG-only model
  - Face-only model
  - Feature-level fusion (FLF)
  - Decision-level fusion (DLF)
- **10-Fold Cross-Validation**: Robust evaluation with statistical measures
- **Modern TensorFlow 2.x**: Updated for compatibility with latest deep learning frameworks
- **Modular Design**: Clean, maintainable code structure following Python best practices
- **Type Hints & Docstrings**: Comprehensive documentation for all functions

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.8+
- NumPy 1.21+
- SciPy 1.7+
- scikit-learn 1.0+

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/soheilrayatdoost/ContinuousEmotionDetection.git
cd ContinuousEmotionDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
ContinuousEmotionDetection/
├── config.py                          # Configuration and hyperparameters
├── data_loader.py                     # Data loading utilities
├── preprocessing.py                   # Data preprocessing and normalization
├── model.py                           # Neural network architectures
├── training.py                        # Training utilities
├── evaluation.py                      # Evaluation metrics
├── cross_validation.py                # Cross-validation utilities
├── main.py                            # Single train/test split (quick testing)
├── main_cv.py                         # 10-fold cross-validation (recommended)
├── requirements.txt                   # Project dependencies
├── README.md                          # This file
├── CROSS_VALIDATION_README.md         # Cross-validation guide
│
└── data/                              # Data directory
    ├── Features/                      # Feature .mat files
    └── lable_continous_Mahnob.mat     # Annotations
```

## 🎯 Usage

### Quick Testing (Single Train/Test Split)

For rapid prototyping and testing (runs in ~10-15 minutes):

```bash
python main.py
```

This uses a single 60/30/10 train/validation/test split.

### Rigorous Evaluation (10-Fold Cross-Validation) ⭐ Recommended

For reliable results and research purposes:

```bash
python main_cv.py
```

This performs 10-fold cross-validation and reports mean ± standard deviation for all metrics. See [`CROSS_VALIDATION_README.md`](CROSS_VALIDATION_README.md) for details.

### Configuration

Edit `config.py` to modify:
- Network hyperparameters (LSTM sizes, dropout rates)
- Training parameters (epochs: 100, batch size: 20)
- File paths

### Custom Training

You can import and use individual modules:

```python
import config
from data_loader import load_data
from preprocessing import normalize_data_subject
from model import network_seq
from training import train_network_seq

# Load your data
features_label, data_num = load_data(config.DATA_FOLDER, annotated_trial)

# Create and train a model
model = network_seq(input_tensor, output_tensor, max_len, 
                   lstm_size1=64, lstm_size2=32, dense_size1=32)
model = train_network_seq(model, train_data, val_data, 
                          label_train, label_val, 
                          epochs=25, batch_size=20)
```

## 📊 Model Architectures

### EEG Model
- Input: 128 EEG band features
- LSTM Layer 1: 64 units
- LSTM Layer 2: 32 units
- Dense Layer: 32 units
- Output: Continuous emotion values

### Face Model
- Input: 38 facial features
- LSTM Layer 1: 19 units
- LSTM Layer 2: 10 units
- Dense Layer: 10 units
- Output: Continuous emotion values

### Feature-Level Fusion (FLF)
- Input: 166 features (38 face + 128 EEG)
- LSTM Layer 1: 83 units
- LSTM Layer 2: 42 units
- Dense Layer: 42 units
- Output: Continuous emotion values

## 📈 Evaluation Metrics

The system evaluates models using:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **Pearson Correlation Coefficient**

### Cross-Validation Results Format

When using `main_cv.py`, results are reported as mean ± standard deviation across 10 folds:

```
EEG:
  MSE     = 0.012345 ± 0.001234
  RMSE    = 0.111111 ± 0.011111
  Pearson = 0.456789 ± 0.045678
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{soleymani2016analysis,
  title={Analysis of EEG Signals and Facial Expressions for Continuous Emotion Detection},
  author={Soleymani, M. and Asghari-Esfeden, S. and Fu, Y. and Pantic, M.},
  journal={IEEE Transactions on Affective Computing},
  volume={7},
  number={1},
  pages={17--28},
  year={2016},
  publisher={IEEE},
  doi={10.1109/TAFFC.2015.2436926}
}
```

**Paper URL**: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7112127&isnumber=7420758

## 🔄 Recent Updates

### December 2025
- ✅ **10-Fold Cross-Validation**: Implemented rigorous CV for reliable evaluation
- ✅ **Increased Training**: Updated to 100 epochs for better convergence
- ✅ **Cross-Validation Module**: Added `cross_validation.py` utility module
- ✅ **Comprehensive Documentation**: Added CV guide and comparison results

### November 2025
- ✅ **Complete Refactoring**: Modular structure with 8 separate modules
- ✅ **TensorFlow 2.x Migration**: Updated from deprecated Keras imports
- ✅ **Type Hints & Docstrings**: Full type annotations and documentation
- ✅ **Bug Fixes**: Fixed global variable issues and deprecated APIs
- ✅ **PEP 8 Compliance**: Consistent snake_case naming conventions
- ✅ **Modern Format**: Using `.keras` model format (TF 2.x standard)
- ✅ **GPU Support**: Added GPU detection and setup guide
- ✅ **Validation**: Compared old vs new implementations for equivalence

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📖 Additional Documentation

- **[CROSS_VALIDATION_README.md](CROSS_VALIDATION_README.md)** - Complete guide to 10-fold CV

## 🎓 Research & Development

### For Development & Testing
Use `main.py` for quick iterations:
- Faster execution (~10-15 minutes)
- Good for debugging and hyperparameter exploration
- Single train/validation/test split

### For Research & Publication
Use `main_cv.py` for final evaluation:
- More reliable results (~3-4 hours)
- 10-fold cross-validation with statistical measures
- Standard practice for machine learning research
- Reports mean ± standard deviation

## 🔍 Code Versions

This repository contains multiple implementations:

1. **`main_cv.py`** ⭐ - Recommended: Refactored code with 10-fold CV
2. **`main.py`** - Refactored code with single split (quick testing)
3. **`ContinousEmotionDetection_cv.py`** - Original code with 10-fold CV
4. **`ContinousEmotionDetection.py`** - Original monolithic code (reference)

All implementations produce functionally equivalent results. See [COMPARISON_RESULTS.md](COMPARISON_RESULTS.md) for validation.

## ⚠️ Important Notes

- **Not Original Paper Code**: This is a reimplementation of the paper's methodology, not the exact code used for the published results
- **Data Format**: Ensure your data files are in the correct format (.mat files)
- **Variable-Length Sequences**: Code handles variable-length sequences using masking
- **Model Format**: Models are saved in `.keras` format (TensorFlow 2.x standard)
- **Normalization**: Subject-wise normalization is applied to features
- **Training Time**: 10-fold CV takes ~3-4 hours on CPU; use GPU for faster training
- **Reproducibility**: Use fixed random seeds for reproducible results
