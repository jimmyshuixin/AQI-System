# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split # Not strictly needed for time series split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import os
import copy
import joblib # 用于保存和加载scaler
import json # For saving model config

import optuna # 用于超参数优化

# --- Matplotlib 中文显示设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败: {e}. 绘图标签可能无法正确显示中文。")

# --- 全局参数设置 (默认值，部分会被配置文件覆盖) ---
DEFAULT_FILE_PATH = '南京_AQI_Data.xlsx' # 默认训练数据文件路径
DEFAULT_LOOK_BACK = 24
DEFAULT_HORIZON = 24
DEFAULT_TARGET_COL_NAME = 'AQI'
DEFAULT_BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 文件保存路径 ---
MODEL_ARTIFACTS_DIR = "D:\code\projects\class\机器学习\AQI\model" # Directory to save all model related files
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model.pth"
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler.pkl"
TARGET_SCALER_SAVE_NAME = "aqi_target_scaler.pkl"
MODEL_CONFIG_SAVE_NAME = "model_config.json"

# --- 训练特定参数 ---
DEFAULT_FULL_TRAIN_EPOCHS = 5  # 完整训练的Epochs(默认值50)
DEFAULT_N_OPTUNA_TRIALS = 5 # Optuna试验次数(默认值50)
DEFAULT_OPTUNA_EPOCHS = 2   # 每个Optuna试验训练的Epochs(默认值15)
DEFAULT_EARLY_STOPPING_PATIENCE = 1 # Early stopping patience (默认值10)
DEFAULT_MIN_DELTA = 0.0001

# --- Utility Functions ---
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def create_sequences(data_df, look_back, horizon, target_col_name, feature_cols, is_predict=False):
    X_list, y_list = [], []
    
    if not all(col in data_df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in data_df.columns]
        raise ValueError(f"Data is missing feature columns for sequence creation: {missing}")
    if not is_predict and target_col_name not in data_df.columns:
        raise ValueError(f"Data is missing target column '{target_col_name}' for sequence creation.")

    data_features_np = data_df[feature_cols].values
    if not is_predict:
        data_target_np = data_df[target_col_name].values

    for i in range(len(data_features_np) - look_back - (0 if is_predict else horizon) + 1):
        input_seq = data_features_np[i:(i + look_back)]
        X_list.append(input_seq)
        if not is_predict:
            output_val = data_target_np[i + look_back : i + look_back + horizon]
            y_list.append(output_val)
    
    X_arr = np.array(X_list) if X_list else np.array([])
    
    if is_predict:
        return X_arr

    y_arr = np.array(y_list) if y_list else np.array([])
    if horizon == 1 and y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    elif horizon > 1 and y_arr.ndim == 1 and y_arr.size > 0:
        y_arr = y_arr.reshape(-1, horizon)
    return X_arr, y_arr

def plot_training_loss(train_losses, val_losses, save_path, title_prefix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{title_prefix}模型训练过程中的损失变化')
    plt.xlabel('Epoch'); plt.ylabel('损失 (MSE)'); plt.legend(); plt.grid(True)
    plt.savefig(save_path)
    plt.close() # Close plot to free memory

def plot_predictions_vs_actual(actual, predicted, save_path, title="AQI实际值 vs. 预测值"):
    plt.figure(figsize=(15, 7))
    # Ensure actual and predicted are 1D for plotting
    actual_flat = actual.flatten() if isinstance(actual, np.ndarray) else actual
    predicted_flat = predicted.flatten() if isinstance(predicted, np.ndarray) else predicted
    
    plt.plot(actual_flat, label='实际AQI', alpha=0.7)
    plt.plot(predicted_flat, label='预测AQI', linestyle='--', alpha=0.7)
    plt.title(title); plt.xlabel('时间步'); plt.ylabel('AQI值'); plt.legend(); plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_anomalies(timestamps_or_indices, actual_values, anomaly_indices, save_path, title="AQI时间序列及检测到的异常点"):
    plt.figure(figsize=(15, 7))
    actual_flat = actual_values.flatten() if isinstance(actual_values, np.ndarray) else actual_values
    plt.plot(timestamps_or_indices, actual_flat, label='实际AQI', alpha=0.7)
    
    valid_anomaly_indices = np.array(anomaly_indices, dtype=int)
    valid_anomaly_indices = valid_anomaly_indices[valid_anomaly_indices < len(actual_flat)]

    if len(valid_anomaly_indices) > 0:
        scatter_x = np.array(timestamps_or_indices)[valid_anomaly_indices]
        scatter_y = np.array(actual_flat)[valid_anomaly_indices]
        plt.scatter(scatter_x, scatter_y, color='red', label='检测到的异常点', marker='o', s=50, zorder=5)
    
    plt.title(title); plt.xlabel('时间/时间步'); plt.ylabel('AQI值'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def detect_anomalies_func(actual_values, predicted_values, factor=1.5):
    residuals = actual_values.flatten() - predicted_values.flatten()
    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - factor * IQR, Q3 + factor * IQR
    anomalies_indices = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
    print(f"异常检测: 残差Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}. 阈值: [{lower_bound:.2f}, {upper_bound:.2f}]. 检测到 {len(anomalies_indices)} 个异常点.")
    return anomalies_indices, residuals, lower_bound, upper_bound

# --- PyTorch Dataset and Model Definition ---
class TimeSeriesDataset(TensorDataset):
    def __init__(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor)

class AQITransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, norm_first=True):
        super(AQITransformer, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, horizon) # Use horizon parameter

    def forward(self, src):
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)
        seq_len = src_embedded.size(1)
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        src_pos_encoded = src_embedded + pe.unsqueeze(0)
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded)
        encoder_output = self.transformer_encoder(src_pos_encoded)
        prediction_input = encoder_output[:, -1, :]
        output = self.output_layer(prediction_input)
        return output

# --- ModelTrainer Class ---
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config['model_artifacts_dir'], exist_ok=True)
        set_seed()

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True, existing_feature_scaler=None, existing_target_scaler=None):
        print(f"开始加载和预处理数据从: {file_path}...")
        try:
            df = pd.read_csv(file_path)
        except Exception:
            try:
                df = pd.read_excel(file_path)
            except Exception as e_excel:
                print(f"读取Excel失败: {e_excel}")
                raise

        if 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Time'])
            df = df.set_index('timestamp').drop(columns=['Time'])
        elif 'date' in df.columns and 'hour' in df.columns:
            df['datetime_str'] = df['date'].astype(str) + df['hour'].astype(str).str.zfill(2)
            df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d%H')
            df = df.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'])
        else:
            print("警告: 未找到标准时间列。尝试将第一列作为时间索引。")
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception:
                raise ValueError("无法自动设置时间索引。请确保数据包含 'Time' 或 ('date' 和 'hour') 列，或第一列是可解析的时间戳。")

        for col in df.columns:
            if col != self.config['target_col_name']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.ffill().bfill().dropna()

        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
            # Add other cyclical features as needed
        
        if self.config['target_col_name'] in df.columns:
            for lag in range(1, self.config['look_back'] // 8 + 1): # Simplified lag creation
                 df[f"{self.config['target_col_name']}_lag_{lag}"] = df[self.config['target_col_name']].shift(lag)
            df = df.dropna()
        else:
             if fit_scalers: # Only raise if target is needed for training/scaling
                raise ValueError(f"目标列 '{self.config['target_col_name']}' 未在数据中找到。")

        if df.empty:
            raise ValueError("数据预处理后DataFrame为空。")

        feature_cols = [col for col in df.columns if col != self.config['target_col_name']]
        
        # Store feature_cols for saving in config later (these are before scaling)
        self.all_feature_columns_for_sequence = feature_cols.copy()


        if fit_scalers:
            feature_scaler = StandardScaler()
            df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
            joblib.dump(feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))
            
            target_scaler = StandardScaler()
            df[[self.config['target_col_name']]] = target_scaler.fit_transform(df[[self.config['target_col_name']]])
            joblib.dump(target_scaler, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALER_SAVE_NAME))
            print("Scalers已拟合和保存。")
            return df, feature_scaler, target_scaler
        else: # Use existing scalers (for prediction preprocessing)
            if existing_feature_scaler is None or (self.config['target_col_name'] in df.columns and existing_target_scaler is None): # Target scaler needed if target is present
                raise ValueError("Existing scalers must be provided if not fitting new ones.")
            
            # Ensure all expected features are present before transform
            expected_features = existing_feature_scaler.feature_names_in_ if hasattr(existing_feature_scaler, 'feature_names_in_') else self.all_feature_columns_for_sequence
            current_features_in_df = [col for col in expected_features if col in df.columns]
            missing_features = [col for col in expected_features if col not in df.columns]
            if missing_features:
                print(f"警告: 预测数据中缺少以下特征，将使用0填充: {missing_features}")
                for mf in missing_features: df[mf] = 0
            
            df[current_features_in_df] = existing_feature_scaler.transform(df[current_features_in_df])

            if self.config['target_col_name'] in df.columns: # If target is present (e.g. for evaluation of prediction data)
                df[[self.config['target_col_name']]] = existing_target_scaler.transform(df[[self.config['target_col_name']]])
            print("Existing scalers applied.")
            return df, existing_feature_scaler, existing_target_scaler


    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, epochs, trial=None):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        train_losses_epoch, val_losses_epoch = [], []

        for epoch in range(epochs):
            model.train()
            running_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * X_batch.size(0)
            epoch_train_loss = running_train_loss / len(train_loader.dataset)
            train_losses_epoch.append(epoch_train_loss)

            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    running_val_loss += loss.item() * X_batch.size(0)
            epoch_val_loss = running_val_loss / len(val_loader.dataset)
            val_losses_epoch.append(epoch_val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

            if trial:
                trial.report(epoch_val_loss, epoch)
                if trial.should_prune():
                    print("Optuna trial pruned.")
                    raise optuna.exceptions.TrialPruned()

            if epoch_val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
                if trial is None: # Only save for final model, not HPO trials
                    torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
                    print(f"Val loss improved. Model saved at epoch {epoch+1}.")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                if best_model_state: model.load_state_dict(best_model_state)
                break
        
        if best_model_state and trial is None and not os.path.exists(os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME)):
            torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
        elif best_model_state:
             model.load_state_dict(best_model_state)
        return model, train_losses_epoch, val_losses_epoch, best_val_loss

    def _objective_optuna(self, trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features):
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        d_model = trial.suggest_categorical('d_model', [64, 128]) # Smaller for faster HPO
        possible_num_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        if not possible_num_heads: raise optuna.exceptions.TrialPruned()
        num_heads = trial.suggest_categorical('num_heads', possible_num_heads)
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 3)
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 4)
        dim_feedforward = d_model * dim_feedforward_factor
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        norm_first = trial.suggest_categorical('norm_first', [True])

        model = AQITransformer(
            num_features=num_input_features, d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, norm_first=norm_first, horizon=self.config['horizon']
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False)
        
        print(f"\nOptuna Trial {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.2f}")
        _, _, _, best_val_loss_trial = self._train_model_core(model, train_loader, val_loader, criterion, optimizer, self.config['optuna_epochs'], trial)
        return best_val_loss_trial

    def run_training_pipeline(self):
        print("--- 开始模型训练流程 ---")
        df_scaled, self.feature_scaler, self.target_scaler = self._load_and_preprocess_data_core(self.config['file_path'], fit_scalers=True)
        
        # After _load_and_preprocess_data_core, self.all_feature_columns_for_sequence is set
        num_input_features_for_model = len(self.all_feature_columns_for_sequence)

        X_initial, y_initial = create_sequences(df_scaled, self.config['look_back'], self.config['horizon'], self.config['target_col_name'], self.all_feature_columns_for_sequence)
        if X_initial.size == 0:
            print("创建序列后数据为空，终止训练。")
            return

        total_samples = X_initial.shape[0]
        train_idx_end = int(total_samples * 0.7) # 70/15/15 split
        val_idx_end = int(total_samples * 0.85)
        X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]
        X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
        X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:]
        print(f"数据集大小: 训练={X_train_np.shape[0]}, 验证={X_val_np.shape[0]}, 测试={X_test_np.shape[0]}")

        if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0:
            print("训练集或验证集为空，无法进行Optuna或训练。")
            return

        print("\n开始Optuna超参数优化...")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: self._objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features_for_model),
                       n_trials=self.config['n_optuna_trials'])
        best_hyperparams = study.best_params
        print(f"最佳试验验证损失: {study.best_value:.6f}\n最佳超参数: {best_hyperparams}")

        print("\n使用最佳超参数训练最终模型...")
        final_model_arch_params = {
            'd_model': best_hyperparams['d_model'], 'nhead': best_hyperparams['num_heads'],
            'num_encoder_layers': best_hyperparams['num_encoder_layers'],
            'dim_feedforward': best_hyperparams['d_model'] * best_hyperparams['dim_feedforward_factor'],
            'dropout': best_hyperparams['dropout_rate'], 'norm_first': best_hyperparams['norm_first']
        }
        final_model = AQITransformer(
            num_features=num_input_features_for_model, **final_model_arch_params, horizon=self.config['horizon']
        ).to(DEVICE)
        
        # Combine train and validation for final training, or use train for training and val for early stopping
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False)

        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_hyperparams['learning_rate'])
        criterion = nn.MSELoss()
        
        final_model, train_losses, val_losses, _ = self._train_model_core(
            final_model, final_train_loader, final_val_loader, criterion, final_optimizer, self.config['full_train_epochs']
        )
        plot_training_loss(train_losses, val_losses, 
                           os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"), 
                           title_prefix="最终模型")
        print(f"最终模型训练完成，并已保存到 {os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME)}")

        # 保存模型配置
        model_config_to_save = {
            'model_architecture': final_model_arch_params,
            'look_back': self.config['look_back'],
            'horizon': self.config['horizon'],
            'target_col_name': self.config['target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence,
            'num_input_features_for_model': num_input_features_for_model
        }
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w') as f:
            json.dump(model_config_to_save, f, indent=4)
        print(f"模型配置已保存到 {os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME)}")

        if X_test_np.shape[0] > 0:
            self.evaluate_trained_model(final_model, X_test_np, y_test_np, self.target_scaler, criterion)

    def evaluate_trained_model(self, model, X_test_np, y_test_np, target_scaler, criterion):
        print("\n评估最终模型在测试集上的性能...")
        test_loader = DataLoader(TimeSeriesDataset(X_test_np, y_test_np), batch_size=self.config['batch_size'], shuffle=False)
        model.eval()
        all_preds_scaled, all_targets_scaled = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(DEVICE))
                all_preds_scaled.append(outputs.cpu().numpy())
                all_targets_scaled.append(y_batch.numpy())
        
        preds_scaled_np = np.concatenate(all_preds_scaled)
        targets_scaled_np = np.concatenate(all_targets_scaled)
        
        actual_orig = target_scaler.inverse_transform(targets_scaled_np)
        predicted_orig = target_scaler.inverse_transform(preds_scaled_np)

        mae = mean_absolute_error(actual_orig, predicted_orig)
        rmse = np.sqrt(mean_squared_error(actual_orig, predicted_orig))
        r2 = r2_score(actual_orig, predicted_orig)
        print(f"测试集评估: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        plot_predictions_vs_actual(actual_orig, predicted_orig, 
                                   os.path.join(self.config['model_artifacts_dir'], "final_model_test_predictions.png"),
                                   title="最终模型测试集: 实际值 vs. 预测值")
        
        anomaly_idx, _, _, _ = detect_anomalies_func(actual_orig, predicted_orig)
        indices_for_plot = np.arange(len(actual_orig.flatten()))
        plot_anomalies(indices_for_plot, actual_orig.flatten(), anomaly_idx, 
                       os.path.join(self.config['model_artifacts_dir'], "final_model_test_anomalies.png"),
                       title="最终模型测试集: 异常点检测")


# --- ModelPredictor Class ---
class ModelPredictor:
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.model_config = None
        self._load_artifacts()

    def _load_artifacts(self):
        print(f"从 {self.artifacts_dir} 加载模型及相关组件...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALER_SAVE_NAME)

        if not all(os.path.exists(p) for p in [config_path, model_path, fs_path, ts_path]):
            raise FileNotFoundError("一个或多个必要的模型文件 (config, model, scalers) 未找到。请先训练模型。")

        with open(config_path, 'r') as f:
            self.model_config = json.load(f)
        
        self.feature_scaler = joblib.load(fs_path)
        self.target_scaler = joblib.load(ts_path)

        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'],
            **self.model_config['model_architecture'],
            horizon=self.model_config['horizon']
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        print("模型及相关组件加载成功。")

    def _preprocess_input_for_prediction(self, df_raw):
        print("预处理输入数据进行预测...")
        # Apply similar preprocessing as in ModelTrainer._load_and_preprocess_data_core (fit_scalers=False part)
        # This is a simplified version, assuming df_raw has a compatible structure for feature creation
        
        # Ensure timestamp index
        if not isinstance(df_raw.index, pd.DatetimeIndex):
            if 'Time' in df_raw.columns:
                df_raw['timestamp'] = pd.to_datetime(df_raw['Time'])
                df_raw = df_raw.set_index('timestamp').drop(columns=['Time'])
            elif 'date' in df_raw.columns and 'hour' in df_raw.columns: # Basic date/hour handling
                 df_raw['datetime_str'] = df_raw['date'].astype(str) + df_raw['hour'].astype(str).str.zfill(2)
                 df_raw['timestamp'] = pd.to_datetime(df_raw['datetime_str'], format='%Y%m%d%H')
                 df_raw = df_raw.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: # Try first column
                try:
                    df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0])
                    df_raw = df_raw.set_index(df_raw.columns[0])
                except Exception as e:
                     raise ValueError(f"无法为预测数据设置时间索引: {e}. 请确保数据包含时间信息。")


        df_processed = df_raw.copy()
        for col in df_processed.columns: # Numeric conversion for features
            if col != self.model_config['target_col_name']: # Target might not be present or needed for scaling here
                 df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Fill NaNs that might result from coerce or be pre-existing
        # Important: Use methods that don't rely on future data for filling if it's a true forecast
        # For simplicity, ffill/bfill. In practice, more sophisticated handling might be needed.
        df_processed = df_processed.ffill().bfill()


        if isinstance(df_processed.index, pd.DatetimeIndex):
            df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed.index.hour / 24.0)
            df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed.index.hour / 24.0)
            # Add other cyclical features consistent with training
        
        # Lag features require the target column to be present in the input df_raw for the look_back period
        if self.model_config['target_col_name'] in df_processed.columns:
            for lag in range(1, self.model_config['look_back'] // 8 + 1): # Match training lag creation
                 df_processed[f"{self.model_config['target_col_name']}_lag_{lag}"] = df_processed[self.model_config['target_col_name']].shift(lag)
            # Drop NaNs created by lags. This means input df_raw must be longer than look_back
            df_processed = df_processed.dropna(subset=[f"{self.model_config['target_col_name']}_lag_{lag}"]) 
        else:
            # If target column is not available, we cannot create target lags.
            # The model was trained with these, so this is a problem.
            # For now, we assume that if lags are in 'all_feature_columns_for_sequence',
            # they must be creatable or already present.
            # A more robust solution would be to check if lag features are in model_config['all_feature_columns_for_sequence']
            # and if self.model_config['target_col_name'] is missing, raise an error or handle it.
            print(f"警告: 目标列 '{self.model_config['target_col_name']}' 未在预测输入数据中找到。无法创建滞后特征。")
            # Check if lag features are expected by the model
            lag_features_expected = [f for f in self.model_config['all_feature_columns_for_sequence'] if '_lag_' in f]
            if lag_features_expected:
                raise ValueError(f"模型期望滞后特征 {lag_features_expected} 但无法从输入数据创建，因为目标列 '{self.model_config['target_col_name']}' 缺失。")


        if df_processed.empty:
            raise ValueError("预测数据预处理后DataFrame为空 (可能由于dropna)。请提供足够长的历史数据。")

        # Scale features
        # Ensure all features expected by the scaler are present
        expected_features_from_scaler = self.feature_scaler.feature_names_in_ if hasattr(self.feature_scaler, 'feature_names_in_') else self.model_config['all_feature_columns_for_sequence']
        
        # Align columns of df_processed to what the scaler expects, adding missing ones with 0
        for col in expected_features_from_scaler:
            if col not in df_processed.columns:
                print(f"警告: 预测数据中缺少特征 '{col}' (缩放器期望)。将使用0填充。")
                df_processed[col] = 0
        
        # Select only the features the scaler was trained on, in the correct order
        features_to_scale_ordered = [col for col in expected_features_from_scaler if col in df_processed.columns] # Should be all of them now
        
        df_processed[features_to_scale_ordered] = self.feature_scaler.transform(df_processed[features_to_scale_ordered])
        
        return df_processed


    def predict(self, input_data_path_or_df):
        if isinstance(input_data_path_or_df, str):
            try:
                df_raw = pd.read_csv(input_data_path_or_df)
            except Exception:
                df_raw = pd.read_excel(input_data_path_or_df)
        elif isinstance(input_data_path_or_df, pd.DataFrame):
            df_raw = input_data_path_or_df.copy()
        else:
            raise ValueError("输入数据必须是文件路径(CSV/Excel)或Pandas DataFrame。")

        if df_raw.shape[0] < self.model_config['look_back']:
             raise ValueError(f"输入数据必须至少包含 {self.model_config['look_back']} 行以满足回看窗口。收到: {df_raw.shape[0]} 行。")


        df_processed = self._preprocess_input_for_prediction(df_raw)
        
        # Use all_feature_columns_for_sequence from loaded config
        X_pred_np = create_sequences(
            df_processed, self.model_config['look_back'], self.model_config['horizon'],
            self.model_config['target_col_name'], # Target name still needed for consistency, though not used for y
            self.model_config['all_feature_columns_for_sequence'],
            is_predict=True
        )

        if X_pred_np.size == 0:
            print("无法从提供的输入数据创建预测序列。")
            return None

        X_pred_torch = torch.from_numpy(X_pred_np).float().to(DEVICE)
        
        with torch.no_grad():
            predictions_scaled = self.model(X_pred_torch)
        
        predictions_original = self.target_scaler.inverse_transform(predictions_scaled.cpu().numpy())
        
        print(f"成功生成 {predictions_original.shape[0]} 条预测，每条包含 {predictions_original.shape[1]} 个时间步。")
        return predictions_original

# --- Main Execution ---
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    
    # Create artifacts directory if it doesn't exist
    os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True)

    action = input("您想做什么？ (1: 训练新模型, 2: 使用现有模型进行预测): ").strip()

    if action == '1':
        train_config = {
            'file_path': input(f"请输入训练数据文件路径 (默认为 '{DEFAULT_FILE_PATH}'): ") or DEFAULT_FILE_PATH,
            'look_back': DEFAULT_LOOK_BACK,
            'horizon': DEFAULT_HORIZON,
            'target_col_name': DEFAULT_TARGET_COL_NAME,
            'batch_size': DEFAULT_BATCH_SIZE,
            'model_artifacts_dir': MODEL_ARTIFACTS_DIR,
            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS,
            'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS,
            'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
        }
        trainer = ModelTrainer(train_config)
        try:
            trainer.run_training_pipeline()
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

    elif action == '2':
        try:
            predictor = ModelPredictor(MODEL_ARTIFACTS_DIR)
            
            input_data_source = input("请输入用于预测的数据文件路径 (CSV 或 Excel): ").strip()
            if not os.path.exists(input_data_source):
                print(f"错误: 预测数据文件 '{input_data_source}' 未找到。")
            else:
                predictions = predictor.predict(input_data_source)
                if predictions is not None:
                    print("\n预测结果 (原始尺度):")
                    # Print first few predictions as an example
                    for i, p_seq in enumerate(predictions[:min(5, len(predictions))]):
                        print(f"  序列 {i+1} ({predictor.model_config['horizon']}步): {np.round(p_seq, 2)}")
                    
                    # Option to save predictions to a file
                    save_pred = input("是否将预测结果保存到CSV文件? (yes/no): ").strip().lower()
                    if save_pred == 'yes':
                        pred_save_path = os.path.join(MODEL_ARTIFACTS_DIR, "predictions_output.csv")
                        # Reshape predictions for easier CSV saving if horizon > 1
                        num_sequences, horizon_steps = predictions.shape
                        pred_df_data = predictions.reshape(num_sequences * horizon_steps, 1) if horizon_steps > 1 else predictions
                        pred_df_columns = [f"{predictor.model_config['target_col_name']}_pred_step_{j+1}" for j in range(horizon_steps)]
                        
                        # Create a more structured DataFrame for multi-step predictions
                        pred_list_of_dicts = []
                        for i in range(num_sequences):
                            row_dict = {'sequence_id': i}
                            for step in range(horizon_steps):
                                row_dict[pred_df_columns[step]] = predictions[i, step]
                            pred_list_of_dicts.append(row_dict)
                        
                        pred_df = pd.DataFrame(pred_list_of_dicts)
                        pred_df.to_csv(pred_save_path, index=False)
                        print(f"预测结果已保存到: {pred_save_path}")

        except FileNotFoundError as e:
            print(f"错误: {e}. 请确保模型已训练并且所有必要文件都在 '{MODEL_ARTIFACTS_DIR}' 目录下。")
        except ValueError as e:
             print(f"预测过程中发生值错误: {e}")
        except Exception as e:
            print(f"预测过程中发生未知错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("无效的选择。请输入 '1' 或 '2'。")

    print("\nAQI预测流程结束。")
    # plt.show() # Uncomment if you want plots to display interactively at the end