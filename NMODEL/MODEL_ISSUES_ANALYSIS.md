# Comprehensive Analysis of Model Issues

## 🔴 Critical Issues

### 1. **TFT Model Producing NaN Predictions from Epoch 1**

**Symptoms:**
- Model outputs NaN immediately after first forward pass
- Training loss becomes NaN from epoch 1
- All batches are skipped due to NaN/Inf warnings

**Root Causes:**

#### A. **Weight Initialization Issues**
- **Problem**: Initial weights may be too small (gain=0.5) or improperly initialized
- **Impact**: Vanishing gradients or unstable initial activations
- **Location**: `tft_model.py` line 293
- **Fix Applied**: Changed gain from 0.1 to 0.5, but may still need adjustment

#### B. **Learning Rate Too High**
- **Problem**: Even with cap at 5e-4, initial gradients may explode
- **Impact**: First update step causes weights to become NaN
- **Location**: `tft_model.py` line 607
- **Current**: `effective_lr = min(self.learning_rate, 5e-4)`
- **Recommendation**: Start with 1e-4 or use learning rate warmup

#### C. **Input Data Scaling Issues**
- **Problem**: StandardScaler may produce extreme values if variance is very small
- **Impact**: Extreme scaled values cause numerical instability
- **Location**: `tft_model.py` lines 477-488
- **Check**: Features with near-zero variance → division by near-zero in scaling

#### D. **Architecture Numerical Instability**
- **Problem**: Multiple operations (LSTM, attention, concatenation) can amplify numerical errors
- **Impact**: Small errors compound into NaN
- **Location**: `tft_model.py` forward pass (lines 325-409)
- **Specific Issues**:
  - LSTM with dropout can produce extreme values
  - Multi-head attention with small sequences may have numerical issues
  - Adding embeddings (X_emb + y_emb) without normalization

#### E. **Gradient Explosion**
- **Problem**: Gradients may explode before clipping
- **Impact**: Weights become NaN after optimizer step
- **Location**: `tft_model.py` line 656
- **Current**: Gradient clipping at 1.0, but may be too late

### 2. **Data Quality Issues**

**Symptoms:**
- Validation errors about NaN/Inf in input data
- Scaling produces NaN values

**Root Causes:**

#### A. **Missing Data Not Handled**
- **Problem**: Input features may contain NaN before scaling
- **Impact**: StandardScaler cannot handle NaN
- **Location**: `utils.py` `prepare_features()` function
- **Check**: Are there missing values in the original CSV files?

#### B. **Infinite Values**
- **Problem**: Division by zero or log(0) in feature engineering
- **Impact**: Inf values propagate through scaling
- **Location**: Feature engineering in `utils.py`
- **Check**: Lag features, rolling means, or mathematical transformations

#### C. **Constant Features**
- **Problem**: Features with zero variance cause division by zero in StandardScaler
- **Impact**: NaN in scaled features
- **Location**: `tft_model.py` `_prepare_data()`
- **Fix Needed**: Remove constant features before scaling

#### D. **Extreme Outliers**
- **Problem**: Very large values cause numerical overflow
- **Impact**: Inf values after operations
- **Location**: All data processing steps
- **Check**: Use robust scaling or clip extreme values

### 3. **Model Architecture Issues**

**Symptoms:**
- NaN appears at specific layers (LSTM, attention, output)
- Model fails even with clean data

**Root Causes:**

#### A. **LSTM Initialization**
- **Problem**: LSTM weights may not be properly initialized
- **Impact**: Unstable hidden states
- **Location**: `tft_model.py` lines 297-302
- **Issue**: Forget gate bias set to 1.0, but other gates may need different initialization

#### B. **Attention Mechanism**
- **Problem**: Multi-head attention with small batch/sequence sizes
- **Impact**: Division by zero in attention scores
- **Location**: `tft_model.py` line 358
- **Check**: Sequence length vs. number of heads

#### C. **Embedding Addition Without Normalization**
- **Problem**: `X_emb = X_emb + y_emb` can cause value explosion
- **Impact**: Values become too large for subsequent layers
- **Location**: `tft_model.py` line 341
- **Fix**: Use layer normalization or scale embeddings

#### D. **Output Layer Architecture**
- **Problem**: Deep output layer (3 linear layers) with ReLU can cause dead neurons
- **Impact**: Zero gradients → NaN weights
- **Location**: `tft_model.py` lines 273-281
- **Check**: Hidden size // 2 may be too small

### 4. **Training Loop Issues**

**Symptoms:**
- Training skips all batches
- Loss never updates
- Model never learns

**Root Causes:**

#### A. **Batch Skipping Logic**
- **Problem**: Skipping batches with NaN means no learning happens
- **Impact**: Model never improves, just keeps producing NaN
- **Location**: `tft_model.py` lines 634-650
- **Issue**: Should stop training early, not skip indefinitely

#### B. **Sample Weight Issues**
- **Problem**: Sample weights may contain NaN or extreme values
- **Impact**: Weighted loss becomes NaN
- **Location**: `tft_model.py` line 645
- **Check**: Validate sample weights before use

#### C. **Empty Dataset**
- **Problem**: TimeSeriesDataset may return 0 samples
- **Impact**: DataLoader is empty, no training happens
- **Location**: `tft_model.py` lines 580-585
- **Check**: Ensure `n_samples > 0` before creating DataLoader

### 5. **Cross-Validation Issues (FIXED)**

**Symptoms:**
- `IndexError: positional indexers are out-of-bounds`
- Wrong data used for validation folds

**Root Causes (FIXED):**
- ✅ Slicing `static_val` instead of `static_train` during CV
- ✅ Incorrect index bounds in CV splits
- ✅ Missing defensive checks

## 🟡 Medium Priority Issues

### 6. **Feature Engineering Issues**

**Potential Problems:**
- Lag features may create NaN at sequence start
- Rolling statistics may have insufficient window
- Cyclical features (sin/cos) should be fine, but check ranges

**Location**: `utils.py` `prepare_features()`

### 7. **Memory Issues**

**Potential Problems:**
- Large batch size with long sequences
- GPU memory overflow
- Location: `tft_model.py` batch_size parameter

### 8. **Hyperparameter Issues**

**Current Settings:**
- `input_window = 72` (3 days of hourly data)
- `output_horizon = 24` (1 day ahead)
- `hidden_size = 96`
- `learning_rate = 5e-4` (capped)
- `batch_size = 128`

**Potential Issues:**
- Input window may be too long for available data
- Hidden size may be too large for the model capacity
- Batch size may be too large for memory

## 🟢 Recommendations for Fixes

### Immediate Actions:

1. **Add Data Validation Before Training**
   ```python
   # Check for constant features
   constant_features = X_train.columns[X_train.nunique() <= 1]
   if len(constant_features) > 0:
       print(f"Removing {len(constant_features)} constant features")
       X_train = X_train.drop(columns=constant_features)
   ```

2. **Use Robust Scaling or Clip Values**
   ```python
   # Instead of StandardScaler, use RobustScaler or clip
   from sklearn.preprocessing import RobustScaler
   # Or clip extreme values before scaling
   X_clipped = np.clip(X.values, -10, 10)
   ```

3. **Add Layer Normalization**
   ```python
   # After embedding addition
   X_emb = self.layer_norm(X_emb + y_emb)
   ```

4. **Reduce Learning Rate Further**
   ```python
   effective_lr = min(self.learning_rate, 1e-4)  # Start even lower
   # Or use warmup
   ```

5. **Check Input Data Quality**
   ```python
   # Before training, print statistics
   print(f"X_train stats: min={X_train.min().min()}, max={X_train.max().max()}")
   print(f"X_train NaN count: {X_train.isna().sum().sum()}")
   print(f"y_train stats: min={y_train.min()}, max={y_train.max()}")
   ```

6. **Simplify Model Architecture**
   - Start with smaller hidden_size (64 instead of 96)
   - Reduce number of layers (1 instead of 2)
   - Simplify output layer (2 layers instead of 3)

7. **Add Early Stopping for NaN**
   ```python
   if torch.any(torch.isnan(pred)):
       print("Stopping training: NaN detected in predictions")
       break  # Stop training, don't just skip batch
   ```

### Diagnostic Steps:

1. **Run Data Quality Check**
   ```python
   # Check for issues in raw data
   train_df = pd.read_csv('F/site3_train.csv')
   print(train_df.describe())
   print(train_df.isna().sum())
   print((train_df == np.inf).sum())
   ```

2. **Test Model with Synthetic Data**
   ```python
   # Create simple synthetic data to test if model architecture works
   X_synth = np.random.randn(1000, 10)
   y_synth = np.random.randn(1000)
   # If this works, issue is in data, not architecture
   ```

3. **Check Scaling Output**
   ```python
   # After scaling, check statistics
   print(f"X_scaled stats: min={X_scaled.min()}, max={X_scaled.max()}")
   print(f"X_scaled NaN: {np.isnan(X_scaled).sum()}")
   ```

4. **Monitor Gradients**
   ```python
   # Before optimizer.step(), check gradients
   total_norm = 0
   for p in self.model.parameters():
       if p.grad is not None:
           param_norm = p.grad.data.norm(2)
           total_norm += param_norm.item() ** 2
   total_norm = total_norm ** (1. / 2)
   print(f"Gradient norm: {total_norm}")
   if total_norm > 10:
       print("WARNING: Large gradients detected")
   ```

## 📊 Summary of Issue Severity

| Issue | Severity | Status | Impact |
|-------|----------|--------|--------|
| TFT NaN from epoch 1 | 🔴 Critical | Active | Model unusable |
| Data scaling NaN | 🔴 Critical | Active | Training fails |
| Architecture instability | 🔴 Critical | Active | Numerical errors |
| CV IndexError | 🟢 Fixed | Resolved | Was blocking training |
| Missing data handling | 🟡 Medium | Needs check | May cause issues |
| Hyperparameter tuning | 🟡 Medium | Needs optimization | Performance impact |
| Memory issues | 🟢 Low | Not observed | May occur with large data |

## 🎯 Priority Fix Order

1. **First**: Check and fix input data quality (NaN, Inf, constant features)
2. **Second**: Simplify model architecture and reduce learning rate
3. **Third**: Add layer normalization and better initialization
4. **Fourth**: Implement robust scaling or value clipping
5. **Fifth**: Add comprehensive logging to identify exact failure point


