# Continuous Emotion Detection

Analysis of EEG Signals and Facial Expressions for Continuous Emotion Detection using LSTM neural networks.

This codebase implements multimodal emotion detection combining EEG signals and facial expressions, with support for feature-level and decision-level fusion strategies.

## 🚀 Features

- **Multimodal Learning**: Combines EEG and facial expression features
- **LSTM-based Architecture**: Handles variable-length time series data
- **Multiple Fusion Strategies**:
  - EEG-only model
  - Face-only model
  - Feature-level fusion (FLF)
  - Decision-level fusion (DLF)
- **Modern TensorFlow 2.x**: Updated for compatibility with latest deep learning frameworks
- **Modular Design**: Clean, maintainable code structure following Python best practices

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
├── config.py              # Configuration and hyperparameters
├── data_loader.py         # Data loading utilities
├── preprocessing.py       # Data preprocessing and normalization
├── model.py              # Neural network architectures
├── training.py           # Training utilities
├── evaluation.py         # Evaluation metrics
├── main.py               # Main execution script
├── requirements.txt      # Project dependencies
└── data/                 # Data directory
    ├── Features/         # Feature .mat files
    └── lable_continous_Mahnob.mat
```

## 🎯 Usage

### Basic Usage

Run the complete pipeline (data loading, training, evaluation):

```bash
python main.py
```

### Configuration

Edit `config.py` to modify:
- Network hyperparameters (LSTM sizes, dropout rates)
- Training parameters (epochs, batch size)
- Data split ratios (train/validation/test)
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

- ✅ Refactored into modular, maintainable code structure
- ✅ Updated to TensorFlow 2.x (from deprecated Keras imports)
- ✅ Added comprehensive type hints and docstrings
- ✅ Fixed bugs and improved error handling
- ✅ Consistent PEP 8 naming conventions (snake_case)
- ✅ Modern `.keras` model format support

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Notes

- Ensure your data files are in the correct format (.mat files)
- The code handles variable-length sequences using masking
- Models are saved in `.keras` format (TensorFlow 2.x standard)
- Subject-wise normalization is applied to features
