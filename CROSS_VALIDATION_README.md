# 10-Fold Cross-Validation Implementation

This directory now includes 10-fold cross-validation versions of both the original and refactored code.

## Files

### Cross-Validation Versions

1. **`main_cv.py`** - Refactored code with 10-fold CV (recommended)
2. **`cross_validation.py`** - Cross-validation utility module (for refactored version)

### Original Single-Split Versions

1. **`main.py`** - Refactored code with single train/val/test split

## Usage

### Run 10-Fold Cross-Validation

```bash
python main_cv.py
```

**Features:**
- Modular structure with separate utility modules
- Type hints and comprehensive docstrings
- Clean output formatting
- Saves weights per fold: `weights_fold{N}_{MODEL}.keras`

**Output:**
- Results for each fold (1-10)
- Aggregated results with mean ± standard deviation
- MSE, RMSE, and Pearson correlation for all models

## What is 10-Fold Cross-Validation?

Cross-validation is a resampling technique that:

1. **Splits data into 10 equal parts (folds)**
2. **Trains on 9 folds, tests on 1 fold** (repeated 10 times)
3. **Each sample is tested exactly once**
4. **Reports average performance** across all folds

### Benefits

- ✅ **More reliable** - Uses all data for both training and testing
- ✅ **Reduces variance** - Multiple train/test splits reduce luck factor
- ✅ **Standard practice** - Common in machine learning research
- ✅ **Better estimates** - Mean ± std deviation of performance

### Process

```
Fold 1: [TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 2: [TRAIN][TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
Fold 3: [TRAIN][TRAIN][TEST][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN]
...
Fold 10: [TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TRAIN][TEST]
```

For each fold:
- 80% of train data → training
- 20% of train data → validation (for early stopping)
- 10% of all data → testing

## Implementation Details

### Data Splitting

```python
# Create 10 folds with fixed random seed for reproducibility
k_folds = create_k_folds(seq_num, k=10, shuffle=True, random_seed=42)

# For each fold
for fold_num, (train_indices, test_indices) in enumerate(k_folds, 1):
    # Split train into train/val (80/20)
    # Train models on train, validate on val, test on test
    # Store results
```

### Result Aggregation

For each model (EEG, Face, FLF, DLF):
- **MSE**: Mean ± Std across 10 folds
- **RMSE**: Mean ± Std across 10 folds  
- **Pearson**: Mean ± Std across 10 folds

## Example Output

```
======================================================================
10-FOLD CROSS-VALIDATION RESULTS (Mean ± Std)
======================================================================

EEG:
  MSE     = 0.012345 ± 0.001234
  RMSE    = 0.111111 ± 0.011111
  Pearson = 0.456789 ± 0.045678

Face:
  MSE     = 0.010234 ± 0.001023
  RMSE    = 0.101234 ± 0.010123
  Pearson = 0.567890 ± 0.056789

Feature Level Fusion:
  MSE     = 0.009123 ± 0.000912
  RMSE    = 0.095456 ± 0.009545
  Pearson = 0.678901 ± 0.067890

Decision Level Fusion:
  MSE     = 0.008912 ± 0.000891
  RMSE    = 0.094345 ± 0.009434
  Pearson = 0.689012 ± 0.068901
```

## Configuration

Edit `config.py` to adjust:
- `EPOCHS` - Number of training epochs (default: 25)
- `BATCH_SIZE` - Batch size for training (default: 20)
- Network hyperparameters (LSTM sizes, dropout rates)

## Saved Models

Cross-validation saves models for each fold:

**Refactored version:**
- `weights_fold1_EEG.keras`, `weights_fold2_EEG.keras`, ...
- `weights_fold1_Face.keras`, `weights_fold2_Face.keras`, ...
- `weights_fold1_FLF.keras`, `weights_fold2_FLF.keras`, ...


## Reproducibility

Both CV implementations use:
- **Fixed random seed (42)** for fold creation
- **Same fold splits** - both implementations use identical train/test splits
- **Shuffled data** - data is shuffled before splitting for randomization

Results should be consistent across runs of the same implementation.

## Comparison: Single Split vs Cross-Validation

| Aspect | Single Split | 10-Fold CV |
|--------|--------------|------------|
| Reliability | Depends on split | Very reliable |
| Data Usage | Some data unused | All data used |
| Results | Single numbers | Mean ± Std |
| Research | Less rigorous | Standard practice |
| Development | Good for testing | Good for evaluation |

**Recommendation:**
- Use **single split** (`main.py`) for development and quick tests
- Use **10-fold CV** (`main_cv.py`) for final evaluation and research papers

## Notes

- Cross-validation uses the same hyperparameters as single-split versions
- All data preprocessing is identical
- Models are independently trained for each fold
- No data leakage between folds
- Validation set is created from training data in each fold
