# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # 虽然时间序列通常顺序划分，但可用于辅助
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import math
import os
import copy
import joblib # 用于保存和加载scaler

import optuna # 用于超参数优化

# 设置matplotlib支持中文显示（如果系统支持SimHei字体）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败: {e}. 绘图标签可能无法正确显示中文。")

# --- 1. 全局参数设置 ---
FILE_PATH = 'model\南京_AQI_Data.xlsx' # 数据文件路径
LOOK_BACK = 24      # 回看窗口大小
HORIZON = 24         # 预测未来24小时
TARGET_COL_NAME = 'AQI' # 目标预测列
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型和Scaler保存路径
MODEL_SAVE_PATH = "best_aqi_transformer_model.pth"
FEATURE_SCALER_SAVE_PATH = "aqi_feature_scaler.pkl"
TARGET_SCALER_SAVE_PATH = "aqi_target_scaler.pkl"

# 初始训练参数 (在Optuna找到最佳参数后，用这些参数进行最终训练)
FULL_TRAIN_EPOCHS = 50 # 用于最终模型训练的Epochs
FULL_TRAIN_LEARNING_RATE = 0.001 # 初始学习率，会被Optuna覆盖

# Optuna 超参数优化相关参数
N_OPTUNA_TRIALS = 50 # Optuna试验次数，可根据计算资源调整
OPTUNA_EPOCHS = 15   # 每个Optuna试验训练的Epochs，应少于完整训练以加速

# 早停参数 (用于完整训练和Optuna试验)
EARLY_STOPPING_PATIENCE = 10
MIN_DELTA = 0.0001

# 在线学习相关参数
ONLINE_LEARNING_ENABLED = False # 是否启用在线学习模拟
ONLINE_FETCH_INTERVAL_SIM = 5 # 模拟每5个“周期”获取一次新数据
ONLINE_FINETUNE_EPOCHS = 5    # 在线微调的Epochs
ONLINE_FINETUNE_LR_FACTOR = 0.1 # 微调时学习率相对于原始学习率的因子

# 设置随机种子以保证结果可复现
def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_seed()

# --- 2. 数据加载与预处理 (与之前类似，但确保scaler被保存和返回) ---
def load_and_preprocess_data(file_path, target_col_name, feature_scaler_path, target_scaler_path, fit_scalers=True):
    """
    加载数据并进行预处理。
    fit_scalers: 是否拟合新的scalers。在初始加载时为True，在线学习加载新数据时为False。
    """
    print("开始加载和预处理数据...")
    try:
        df = pd.read_csv(file_path)
    except Exception:
        try:
            df = pd.read_excel(file_path)
        except Exception as e_excel:
            print(f"作为Excel也读取失败: {e_excel}")
            raise

    print("\n数据初步信息:")
    df.info()
    
    if 'date' in df.columns and 'hour' in df.columns:
        try:
            df['datetime_str'] = df['date'].astype(str) + df['hour'].astype(str).str.zfill(2)
            df['timestamp'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d%H')
            df = df.set_index('timestamp')
            df = df.drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
        except Exception as e:
            print(f"处理date和hour列时出错: {e}. 假设第一列是时间戳（如果适用）。")
            if pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]):
                 df = df.set_index(df.columns[0])
            else:
                 print("警告：未能自动设置时间索引。")
    else:
        print("警告：数据中未找到 'date' 和 'hour' 列。尝试使用第一列作为时间索引。")
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])
        except Exception:
            print("转换第一列为时间戳失败。")


    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.ffill().bfill()
    if df.isnull().sum().sum() > 0:
        print("警告：填充后仍有NaN值。")
        df = df.dropna()


    if isinstance(df.index, pd.DatetimeIndex):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    else:
        print("警告：索引不是DatetimeIndex，无法创建周期性时间特征。")

    if target_col_name in df.columns:
        for lag in range(1, 4):
            df[f'{target_col_name}_lag_{lag}'] = df[target_col_name].shift(lag)
        df = df.dropna()
    else:
        raise ValueError(f"目标列 '{target_col_name}' 未在数据中找到。")

    if df.empty:
        raise ValueError("数据预处理后DataFrame为空。")

    feature_cols = [col for col in df.columns if col != target_col_name]
    
    if fit_scalers:
        feature_scaler = StandardScaler()
        df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
        joblib.dump(feature_scaler, feature_scaler_path)
        
        target_scaler = StandardScaler()
        df[[target_col_name]] = target_scaler.fit_transform(df[[target_col_name]])
        joblib.dump(target_scaler, target_scaler_path)
        print("Scalers已拟合和保存。")
    else:
        try:
            feature_scaler = joblib.load(feature_scaler_path)
            df[feature_cols] = feature_scaler.transform(df[feature_cols])
            
            target_scaler = joblib.load(target_scaler_path)
            df[[target_col_name]] = target_scaler.transform(df[[target_col_name]])
            print("Scalers已加载和应用。")
        except FileNotFoundError:
            raise FileNotFoundError("Scaler文件未找到。请先运行初始训练以拟合和保存scalers。")
            
    print("数据预处理完成。")
    return df, feature_scaler, target_scaler

# --- 3. 创建序列 (与之前相同) ---
def create_sequences(data, look_back, horizon, target_col_name, feature_cols):
    X, y_vals = [], []
    data_features_np = data[feature_cols].values
    data_target_np = data[target_col_name].values

    for i in range(len(data_features_np) - look_back - horizon + 1):
        input_seq = data_features_np[i:(i + look_back)]
        output_val = data_target_np[i + look_back : i + look_back + horizon]
        X.append(input_seq)
        y_vals.append(output_val)
    
    X_arr = np.array(X)
    y_arr = np.array(y_vals)
    
    if horizon == 1 and y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    elif horizon > 1 and y_arr.ndim == 1: # 如果horizon > 1但结果是1D，可能需要调整
        y_arr = y_arr.reshape(-1, horizon)

    return X_arr, y_arr

# --- 4. PyTorch Dataset (与之前相同) ---
class TimeSeriesDataset(TensorDataset):
    def __init__(self, X, y):
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor)

# --- 5. Transformer 模型架构 (与之前相同，但接受超参数) ---
class AQITransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, norm_first=True):
        super(AQITransformer, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # 简化的位置编码直接在forward中实现
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, HORIZON)

    def forward(self, src):
        # src shape: (batch_size, seq_len, num_features)
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # 简化的位置编码 (直接添加)
        # batch_first=True, src_embedded is (N, S, E)
        seq_len = src_embedded.size(1)
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 != 0: # 奇数d_model处理
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        src_pos_encoded = src_embedded + pe.unsqueeze(0) # (1, seq_len, d_model) broadcasts
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded) # Dropout after PE, hardcoded for now or pass as param

        encoder_output = self.transformer_encoder(src_pos_encoded)
        prediction_input = encoder_output[:, -1, :]
        output = self.output_layer(prediction_input)
        return output

# --- 6. 模型训练函数 (修改以支持Optuna剪枝和早停) ---
def train_model_core(model, train_loader, val_loader, criterion, optimizer, epochs, device, 
                     model_save_path=None, trial=None, early_stopping_patience=10, min_delta=0.0001):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses_epoch = []
    val_losses_epoch = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_val_loss += loss.item() * X_batch.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses_epoch.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        if trial: # Optuna剪枝逻辑
            trial.report(epoch_val_loss, epoch)
            if trial.should_prune():
                print("Optuna trial pruned.")
                raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            if model_save_path: # 仅在完整训练时保存
                torch.save(best_model_state, model_save_path)
                print(f"Val loss improved. Model saved at epoch {epoch+1}.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            if best_model_state: model.load_state_dict(best_model_state)
            break
            
    if best_model_state and model_save_path: # 确保加载最佳模型 (如果早停发生)
         model.load_state_dict(best_model_state)
    
    return model, train_losses_epoch, val_losses_epoch, best_val_loss


# --- 7. Optuna 目标函数 ---
def objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features, device):
    # 定义超参数搜索空间 (参考文档表1和第13.3节)
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    d_model = trial.suggest_categorical('d_model', [64, 128, 256]) # 调整为文档建议
    
    # 确保 d_model % num_heads == 0
    possible_num_heads = [h for h in [2, 4, 8] if d_model % h == 0] # 调整num_heads选项
    if not possible_num_heads: # 如果没有合适的头数，剪枝
        print(f"d_model={d_model} 与可选头数不兼容，剪枝。")
        raise optuna.exceptions.TrialPruned()
    num_heads = trial.suggest_categorical('num_heads', possible_num_heads)

    num_encoder_layers = trial.suggest_int('num_encoder_layers', 1, 4) # 文档建议1-6，代码中1-4
    
    # dim_feedforward 通常是 d_model 的倍数
    dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 4)
    dim_feedforward = d_model * dim_feedforward_factor
    
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3) # 文档建议0.1-0.3
    norm_first = trial.suggest_categorical('norm_first', [True]) # 文档推荐True

    model = AQITransformer(
        num_features=num_input_features,
        d_model=d_model,
        nhead=num_heads,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout_rate,
        norm_first=norm_first
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_dataset_hpo = TimeSeriesDataset(X_train_np, y_train_np)
    val_dataset_hpo = TimeSeriesDataset(X_val_np, y_val_np)
    train_loader_hpo = DataLoader(train_dataset_hpo, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_hpo = DataLoader(val_dataset_hpo, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nOptuna Trial {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.2f}")

    _, _, _, best_val_loss_trial = train_model_core(
        model, train_loader_hpo, val_loader_hpo, criterion, optimizer, 
        epochs=OPTUNA_EPOCHS, device=device, trial=trial,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA
    )
    
    return best_val_loss_trial # Optuna会最小化这个值

# --- 8. 模型评估 (与之前相同) ---
def evaluate_model(model, test_loader, criterion, target_scaler, device):
    model.eval()
    all_preds, all_targets = [], []
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
            
    test_loss /= len(test_loader.dataset)
    predictions_scaled = np.concatenate(all_preds, axis=0)
    targets_scaled = np.concatenate(all_targets, axis=0)
    
    predictions_original = target_scaler.inverse_transform(predictions_scaled)
    targets_original = target_scaler.inverse_transform(targets_scaled)

    mae = mean_absolute_error(targets_original, predictions_original)
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets_original, predictions_original)
    print(f"测试集评估: Loss={test_loss:.6f}, MAE={mae:.4f}, MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return targets_original, predictions_original, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# --- 9. 异常检测 (与之前相同) ---
def detect_anomalies(actual_values, predicted_values, factor=1.5):
    residuals = actual_values.flatten() - predicted_values.flatten()
    Q1, Q3 = np.percentile(residuals, [25, 75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - factor * IQR, Q3 + factor * IQR
    anomalies_indices = np.where((residuals < lower_bound) | (residuals > upper_bound))[0]
    print(f"异常检测: 残差Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}. 阈值: [{lower_bound:.2f}, {upper_bound:.2f}]. 检测到 {len(anomalies_indices)} 个异常点.")
    return anomalies_indices, residuals, lower_bound, upper_bound

# --- 10. 结果可视化 (与之前相同, 略作简化) ---
def plot_training_loss(train_losses, val_losses, title_prefix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{title_prefix}模型训练过程中的损失变化')
    plt.xlabel('Epoch'); plt.ylabel('损失 (MSE)'); plt.legend(); plt.grid(True)
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}_training_loss.png")
    plt.show()

def plot_predictions(actual, predicted, title="AQI实际值 vs. 预测值 (测试集)"):
    plt.figure(figsize=(15, 7))
    plt.plot(actual, label='实际AQI', alpha=0.7)
    plt.plot(predicted, label='预测AQI', linestyle='--', alpha=0.7)
    plt.title(title); plt.xlabel('时间步'); plt.ylabel('AQI值'); plt.legend(); plt.grid(True)
    plt.savefig("predictions_vs_actual.png")
    plt.show()

def plot_anomalies(timestamps_or_indices, actual_values, anomaly_indices, title="AQI时间序列及检测到的异常点"):
    plt.figure(figsize=(15, 7))
    plt.plot(timestamps_or_indices, actual_values, label='实际AQI', alpha=0.7)
    valid_anomaly_indices = [idx for idx in anomaly_indices if idx < len(actual_values)]
    if valid_anomaly_indices:
        anomaly_display_indices = np.array(timestamps_or_indices)[valid_anomaly_indices] if isinstance(timestamps_or_indices, (list, np.ndarray)) else timestamps_or_indices[valid_anomaly_indices]
        plt.scatter(anomaly_display_indices, actual_values[valid_anomaly_indices], 
                    color='red', label='检测到的异常点', marker='o', s=50, zorder=5)
    plt.title(title); plt.xlabel('时间/时间步'); plt.ylabel('AQI值'); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("aqi_anomalies_visualization.png")
    plt.show()


# --- 11. 在线学习模块 ---
def fetch_new_data_online_placeholder(df_columns, num_rows=LOOK_BACK + HORIZON + 5):
    """
    模拟从在线来源获取新数据。
    实际应用中，这里会包含网页抓取或API调用逻辑。
    返回一个与原始数据结构相似的DataFrame。
    """
    print("\n模拟获取在线新数据...")
    # 生成一些随机数据作为占位符，结构与原始数据相似
    # 确保生成的随机数据范围大致合理，或者使用部分历史数据模拟
    data = {}
    for col in df_columns:
        if 'sin' in col or 'cos' in col: # 周期特征在-1到1之间
            data[col] = np.random.uniform(-1, 1, num_rows)
        elif TARGET_COL_NAME in col: # AQI值
            data[col] = np.random.uniform(10, 300, num_rows)
        else: # 其他污染物
            data[col] = np.random.uniform(0, 150, num_rows)
    
    # 创建一个未来的时间索引
    # 假设最后一个已知时间戳是 df_scaled.index[-1] (这需要在主程序中传递或获取)
    # 为简化，这里只返回数据，不处理时间戳对齐
    new_df = pd.DataFrame(data)
    print(f"成功模拟获取 {len(new_df)} 条新数据。")
    return new_df

def online_fine_tune_model(model, new_data_df, feature_scaler, target_scaler, 
                           look_back, horizon, target_col_name, all_feature_cols,
                           finetune_epochs, base_lr, lr_factor, device):
    """
    使用新数据对现有模型进行微调。
    """
    print("开始在线微调模型...")
    
    # 1. 预处理新数据 (使用已加载的scalers)
    new_data_df_processed = new_data_df.copy()
    feature_cols_in_new_data = [col for col in new_data_df_processed.columns if col != target_col_name and col in all_feature_cols]

    # 确保新数据中包含所有必要的特征列
    missing_cols = [col for col in feature_cols_in_new_data if col not in new_data_df_processed.columns]
    if missing_cols:
        print(f"警告: 新数据中缺少以下特征列: {missing_cols}。将尝试使用0填充。")
        for mc in missing_cols: new_data_df_processed[mc] = 0


    new_data_df_processed[feature_cols_in_new_data] = feature_scaler.transform(new_data_df_processed[feature_cols_in_new_data])
    new_data_df_processed[[target_col_name]] = target_scaler.transform(new_data_df_processed[[target_col_name]])
    
    # 2. 创建序列
    X_new, y_new = create_sequences(new_data_df_processed, look_back, horizon, target_col_name, all_feature_cols)
    
    if X_new.shape[0] == 0:
        print("新数据不足以创建序列进行微调，跳过。")
        return model, [], []

    new_dataset = TimeSeriesDataset(X_new, y_new)
    new_data_loader = DataLoader(new_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. 设置微调优化器 (使用较小的学习率)
    finetune_lr = base_lr * lr_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    criterion = nn.MSELoss()
    
    print(f"微调模型 {finetune_epochs} epochs, 学习率: {finetune_lr:.7f}")
    
    # 4. 微调 (使用 train_model_core，但不保存模型，也没有trial)
    # 注意：在线微调通常没有独立的验证集，或者使用一小部分新数据作为临时验证
    # 这里简化，直接在所有新数据上训练
    model.train()
    finetune_train_losses = []
    for epoch in range(finetune_epochs):
        running_loss = 0.0
        for X_batch, y_batch in new_data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        epoch_loss = running_loss / len(new_data_loader.dataset)
        finetune_train_losses.append(epoch_loss)
        print(f"在线微调 Epoch [{epoch+1}/{finetune_epochs}], 损失: {epoch_loss:.6f}")
        
    print("在线微调完成。")
    return model, finetune_train_losses, [] # 返回空的val_losses


# --- 主程序 ---
if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")

    # 1. 初始数据加载和预处理
    try:
        df_scaled_initial, fs_initial, ts_initial = load_and_preprocess_data(
            FILE_PATH, TARGET_COL_NAME, 
            FEATURE_SCALER_SAVE_PATH, TARGET_SCALER_SAVE_PATH, 
            fit_scalers=True
        )
    except Exception as e:
        print(f"初始数据加载或预处理失败: {e}")
        exit()

    if df_scaled_initial.empty:
        print("初始预处理后的数据为空，程序终止。")
        exit()
        
    all_feature_columns = df_scaled_initial.columns.tolist()
    num_input_features = len(all_feature_columns) # X.shape[2] is num_features in sequence

    # 2. 创建初始序列
    X_initial, y_initial = create_sequences(df_scaled_initial, LOOK_BACK, HORIZON, TARGET_COL_NAME, all_feature_columns)
    if X_initial.size == 0:
        print("创建初始序列后数据为空，程序终止。")
        exit()

    # 3. 数据集划分 (用于Optuna和最终训练)
    # 严格按时间顺序划分：60% 训练, 20% 验证 (用于Optuna和早停), 20% 测试
    total_samples = X_initial.shape[0]
    train_idx_end = int(total_samples * 0.6)
    val_idx_end = int(total_samples * 0.8)

    X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]
    X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
    X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:]
    
    print(f"初始数据集大小: 训练集={X_train_np.shape[0]}, 验证集={X_val_np.shape[0]}, 测试集={X_test_np.shape[0]}")

    if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0:
        print("训练集或验证集为空，无法进行Optuna或训练。程序终止。")
        exit()

    # 4. Optuna 超参数优化
    print("\n开始Optuna超参数优化...")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features, DEVICE), 
                   n_trials=N_OPTUNA_TRIALS)

    print("\nOptuna超参数优化完成。")
    best_hyperparams = study.best_params
    print(f"最佳试验的验证损失: {study.best_value:.6f}")
    print("最佳超参数:")
    for key, value in best_hyperparams.items():
        print(f"  {key}: {value}")

    # 5. 使用最佳超参数训练最终模型
    print("\n使用Optuna找到的最佳超参数训练最终模型...")
    
    # 从最佳试验中提取参数
    final_lr = best_hyperparams['learning_rate']
    final_d_model = best_hyperparams['d_model']
    final_num_heads = best_hyperparams['num_heads']
    final_num_encoder_layers = best_hyperparams['num_encoder_layers']
    final_dim_feedforward = final_d_model * best_hyperparams['dim_feedforward_factor']
    final_dropout_rate = best_hyperparams['dropout_rate']
    final_norm_first = best_hyperparams['norm_first']

    final_model = AQITransformer(
        num_features=num_input_features,
        d_model=final_d_model,
        nhead=final_num_heads,
        num_encoder_layers=final_num_encoder_layers,
        dim_feedforward=final_dim_feedforward,
        dropout=final_dropout_rate,
        norm_first=final_norm_first
    ).to(DEVICE)
    
    # 合并训练集和验证集进行最终训练，或仅在训练集上训练，然后在测试集上评估
    # 这里选择在 (训练集+验证集) 上训练，并使用早停 (基于一部分数据作为临时验证或不使用早停)
    # 为简单起见，仍在原训练集和验证集上训练，但使用完整epochs
    
    # 创建用于最终训练的数据加载器
    # 可以选择合并 X_train_np 和 X_val_np 进行最终训练，然后在 X_test_np 上测试
    # 或者，为了简单和利用早停，仍然使用原始的 train/val 划分
    final_train_dataset = TimeSeriesDataset(X_train_np, y_train_np)
    final_val_dataset = TimeSeriesDataset(X_val_np, y_val_np) # 用于早停
    final_train_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=final_lr)
    final_criterion = nn.MSELoss()

    final_model, final_train_losses, final_val_losses, _ = train_model_core(
        final_model, final_train_loader, final_val_loader, final_criterion, final_optimizer,
        epochs=FULL_TRAIN_EPOCHS, device=DEVICE, model_save_path=MODEL_SAVE_PATH,
        early_stopping_patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA
    )
    plot_training_loss(final_train_losses, final_val_losses, title_prefix="最终模型")
    print(f"最终模型训练完成，并已保存到 {MODEL_SAVE_PATH}")

    # 6. 评估最终模型
    if X_test_np.shape[0] > 0:
        print("\n评估最终模型在测试集上的性能...")
        # 加载保存的最佳模型 (train_model_core 内部会加载最佳状态)
        # final_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        
        test_dataset = TimeSeriesDataset(X_test_np, y_test_np)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        actual_orig, predicted_orig, metrics = evaluate_model(final_model, test_loader, final_criterion, ts_initial, DEVICE)
        plot_predictions(actual_orig.flatten(), predicted_orig.flatten())

        # 7. 异常检测
        print("\n在测试集上进行异常检测...")
        anomaly_idx, residuals, lower_b, upper_b = detect_anomalies(actual_orig, predicted_orig)
        test_indices = np.arange(len(actual_orig)) # 使用数字索引绘图
        plot_anomalies(test_indices, actual_orig.flatten(), anomaly_idx)
    else:
        print("测试集为空，跳过最终评估和异常检测。")

    # 8. 模拟在线学习过程
    if ONLINE_LEARNING_ENABLED:
        print("\n--- 开始模拟在线学习过程 ---")
        current_model_for_online = final_model # 从训练好的最终模型开始
        base_lr_for_online = final_lr # 使用最终训练的学习率作为微调的基础

        # 加载scalers，因为在线学习函数需要它们
        try:
            feature_scaler_online = joblib.load(FEATURE_SCALER_SAVE_PATH)
            target_scaler_online = joblib.load(TARGET_SCALER_SAVE_PATH)
        except FileNotFoundError:
            print("Scaler文件未找到，无法进行在线学习。")
            ONLINE_LEARNING_ENABLED = False

        if ONLINE_LEARNING_ENABLED:
            for online_step in range(1, ONLINE_FETCH_INTERVAL_SIM * 3 + 1): # 模拟几个更新周期
                print(f"\n在线学习周期 {online_step}")
                
                # 模拟获取新数据
                # 注意：df_scaled_initial.columns 包含了所有特征名，包括原始、周期、滞后
                # all_feature_columns 已经定义为 df_scaled_initial.columns.tolist()
                new_online_data_df = fetch_new_data_online_placeholder(df_scaled_initial.columns, num_rows=LOOK_BACK + HORIZON + 20)
                
                # 在线微调
                current_model_for_online, ft_losses, _ = online_fine_tune_model(
                    current_model_for_online, new_online_data_df, 
                    feature_scaler_online, target_scaler_online,
                    LOOK_BACK, HORIZON, TARGET_COL_NAME, all_feature_columns,
                    ONLINE_FINETUNE_EPOCHS, base_lr_for_online, ONLINE_FINETUNE_LR_FACTOR, DEVICE
                )
                if ft_losses:
                    plot_training_loss(ft_losses, [], title_prefix=f"在线微调周期{online_step}_")

                # (可选) 在线评估：可以使用一部分新获取的数据作为临时测试集
                # 这里为了简单，我们可以在固定的原始测试集上再次评估，观察性能变化
                # 或者，如果 fetch_new_data_online_placeholder 能提供带标签的测试数据，则更佳
                if X_test_np.shape[0] > 0:
                    print(f"在线学习周期 {online_step} 后，在原始测试集上重新评估模型:")
                    test_dataset_online_eval = TimeSeriesDataset(X_test_np, y_test_np) # 使用原始测试集
                    test_loader_online_eval = DataLoader(test_dataset_online_eval, batch_size=BATCH_SIZE, shuffle=False)
                    evaluate_model(current_model_for_online, test_loader_online_eval, final_criterion, ts_initial, DEVICE)
                
                if online_step % ONLINE_FETCH_INTERVAL_SIM == 0:
                    print(f"--- 在线学习模拟 {online_step // ONLINE_FETCH_INTERVAL_SIM} 大周期结束 ---")
    
    print("\nAQI预测与异常检测流程结束。")

