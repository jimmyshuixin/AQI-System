# ==============================================================================
# 核心功能:
# 该脚本构建了一个综合系统，利用Transformer神经网络模型对多种空气质量指标（包括AQI和各项污染物浓度）
# 进行多目标预测，并集成了异常数据检测功能。
# ... (其他注释保持不变)
# ==============================================================================

# 导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau # 学习率调度器

import math
import os
import copy
import joblib # 用于保存和加载scaler对象
import json # 用于保存和加载JSON格式的模型配置

import optuna # 自动化超参数优化库

# --- Matplotlib 中文显示全局设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"警告: Matplotlib中文字体设置失败: {e}。图表中的中文标签可能无法正常显示。")

# --- 全局核心参数定义 ---
# 使用正斜杠避免路径中的转义序列警告
DEFAULT_FILE_PATH = 'data_process/output/南京_AQI_Data.xlsx' # 默认训练数据文件的相对或绝对路径
DEFAULT_LOOK_BACK = 24 # 模型回溯历史数据的时间窗口长度（单位：小时） # 用户应重点检查此值！
DEFAULT_HORIZON = 72   # 模型向前预测的时间范围（单位：小时）
DEFAULT_TARGET_COL_NAMES = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
DEFAULT_ALL_AVAILABLE_COL_NAMES = [ 
    'date', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
    'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 
    'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'
]
DEFAULT_PRIMARY_TARGET_COL_NAME = 'AQI' 
DEFAULT_BATCH_SIZE = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# 使用正斜杠避免路径中的转义序列警告
MODEL_ARTIFACTS_DIR = "model/output" # 存储所有模型相关文件（权重、配置、缩放器等）的目录名
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model_adv.pth" 
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler_adv.pkl" 
TARGET_SCALERS_SAVE_NAME = "aqi_target_scalers_adv.pkl" 
MODEL_CONFIG_SAVE_NAME = "model_config_adv.json" 

DEFAULT_FULL_TRAIN_EPOCHS = 200  
DEFAULT_N_OPTUNA_TRIALS = 150    
DEFAULT_OPTUNA_EPOCHS = 30     
DEFAULT_EARLY_STOPPING_PATIENCE = 20 
DEFAULT_MIN_DELTA = 0.00001 
DEFAULT_ANOMALY_THRESHOLD_FACTOR = 3.0 

IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
POLLUTANT_BREAKPOINTS = {
    'SO2_24h':   [0, 50,  150,  475,  800,  1600, 2100, 2620], 
    'NO2_24h':   [0, 40,  80,   180,  280,  565,  750,  940],  
    'PM10_24h':  [0, 50,  150,  250,  350,  420,  500,  600],  
    'CO_24h':    [0, 2,   4,    14,   24,   36,   48,   60],   
    'O3_8h_24h': [0, 100, 160,  215,  265,  800, 1000, 1200],  
    'O3_1h':     [0, 160, 200,  300,  400,  800, 1000, 1200],  
    'PM2.5_24h': [0, 35,  75,   115,  150,  250,  350,  500]   
}
POLLUTANT_BREAKPOINTS_HOURLY_APPROX = {
    'SO2':   [0, 150, 500,  650,  800], 
    'NO2':   [0, 100, 200,  700, 1200], 
    'PM10':  [0, 50, 150, 250, 350, 420], 
    'CO':    [0, 5,  10,   35,   60],    
    'O3':    POLLUTANT_BREAKPOINTS['O3_1h'], 
    'PM2.5': [0, 35, 75, 115, 150, 250] 
}

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def calculate_iaqi(Cp, pollutant_key):
    if pd.isna(Cp) or Cp < 0: 
        return np.nan 
    bp_table_to_use = POLLUTANT_BREAKPOINTS 
    if pollutant_key not in bp_table_to_use:
        bp_table_to_use = POLLUTANT_BREAKPOINTS_HOURLY_APPROX 
        if pollutant_key not in bp_table_to_use:
            return np.nan
    bp = bp_table_to_use.get(pollutant_key) 
    if Cp > bp[-1]: 
        return 500 
    for i in range(len(bp) - 1):
        if bp[i] <= Cp < bp[i+1]: 
            IAQI_Lo, IAQI_Hi = IAQI_LEVELS[i], IAQI_LEVELS[i+1]
            BP_Lo, BP_Hi = bp[i], bp[i+1]
            if BP_Hi == BP_Lo: return IAQI_Lo 
            return round(((IAQI_Hi - IAQI_Lo) / (BP_Hi - BP_Lo)) * (Cp - BP_Lo) + IAQI_Lo)
    if Cp == bp[0]: 
        return IAQI_LEVELS[0]
    return np.nan 

def calculate_aqi_from_pollutants(df):
    iaqi_df = pd.DataFrame(index=df.index) 
    pollutants_for_calc = {
        'SO2_24h':   ['SO2_24h', 'SO2'], 
        'NO2_24h':   ['NO2_24h', 'NO2'],
        'PM10_24h':  ['PM10_24h', 'PM10'], 
        'CO_24h':    ['CO_24h', 'CO'],
        'O3_8h_24h': ['O3_8h_24h', 'O3_8h', 'O3'], 
        'PM2.5_24h': ['PM2.5_24h', 'PM2.5']
    }
    for bp_key, df_col_options in pollutants_for_calc.items():
        selected_col_for_iaqi = None
        for df_col in df_col_options: 
            if df_col in df.columns and not df[df_col].isnull().all():
                selected_col_for_iaqi = df_col
                break 
        if selected_col_for_iaqi:
            iaqi_df[bp_key] = df[selected_col_for_iaqi].apply(lambda x: calculate_iaqi(x, bp_key if bp_key in POLLUTANT_BREAKPOINTS else selected_col_for_iaqi))
        else:
            iaqi_df[bp_key] = np.nan 
    df['AQI_calculated'] = iaqi_df.max(axis=1, skipna=True)
    def get_primary_pollutants(row):
        if pd.isna(row['AQI_calculated']) or row['AQI_calculated'] <= 50: return '无' 
        primary = [pollutant_bp_key for pollutant_bp_key in iaqi_df.columns 
                   if pd.notna(row[pollutant_bp_key]) and round(row[pollutant_bp_key]) == round(row['AQI_calculated'])]
        return ', '.join(primary) if primary else '无'
    temp_iaqi_df_for_primary = iaqi_df.copy()
    temp_iaqi_df_for_primary['AQI_calculated'] = df['AQI_calculated']
    df['Primary_Pollutant_calculated'] = temp_iaqi_df_for_primary.apply(get_primary_pollutants, axis=1)
    return df

def create_sequences(data_df, look_back, horizon, target_col_names, feature_cols, is_predict=False):
    X_list, y_list = [], []
    missing_feature_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_feature_cols: raise ValueError(f"数据DataFrame中缺少必要的特征列: {missing_feature_cols}.")
    if not is_predict:
        missing_target_cols = [col for col in target_col_names if col not in data_df.columns]
        if missing_target_cols: raise ValueError(f"数据DataFrame中缺少必要的目标列: {missing_target_cols}.")

    data_features_np = data_df[feature_cols].values
    if not is_predict: data_targets_np = data_df[target_col_names].values
    
    num_samples = len(data_features_np)
    if is_predict: 
        num_possible_sequences = num_samples - look_back + 1
    else: 
        num_possible_sequences = num_samples - look_back - horizon + 1

    if num_possible_sequences <= 0: 
        num_features = len(feature_cols)
        num_targets = len(target_col_names) if not is_predict else 0
        empty_x_shape = (0, look_back, num_features)
        empty_y_shape = (0, horizon, num_targets) if not is_predict else (0,)
        # 使用 float32
        return np.empty(empty_x_shape, dtype=np.float32), (np.empty(empty_y_shape, dtype=np.float32) if not is_predict else None)

    for i in range(num_possible_sequences):
        X_list.append(data_features_np[i : i + look_back])
        if not is_predict:
            y_list.append(data_targets_np[i + look_back : i + look_back + horizon, :])
            
    # 创建NumPy数组时指定dtype为float32以减少内存使用
    X_arr = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, look_back, len(feature_cols)), dtype=np.float32)
    
    if is_predict: 
        return X_arr, None
        
    y_arr = np.array(y_list, dtype=np.float32) if y_list else np.empty((0, horizon, len(target_col_names)), dtype=np.float32)
    
    if y_arr.size == 0 and X_arr.size > 0 and not is_predict : 
        print("警告: 已创建输入(X)序列，但由于数据末端长度不足，未能生成对应的输出(y)序列。")
        return np.empty((0, look_back, len(feature_cols)), dtype=np.float32), np.empty((0, horizon, len(target_col_names)), dtype=np.float32)
        
    return X_arr, y_arr

def plot_training_loss(train_losses, val_losses, save_path, title_prefix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失 (Train Loss)')
    plt.plot(val_losses, label='验证损失 (Validation Loss)')
    plt.title(f'{title_prefix}模型训练过程中的损失变化')
    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('损失函数值 (Loss - MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_predictions_vs_actual(actual, predicted, target_name, save_path_prefix, title_suffix="实际值 vs. 预测值"):
    plt.figure(figsize=(15, 7))
    actual_flat = actual.flatten() 
    predicted_flat = predicted.flatten()
    plt.plot(actual_flat, label=f'实际值 ({target_name})', alpha=0.7)
    plt.plot(predicted_flat, label=f'预测值 ({target_name})', linestyle='--', alpha=0.7)
    plt.title(f'{target_name} - {title_suffix}')
    plt.xlabel('时间步 (Time Step)')
    plt.ylabel(f'{target_name} 浓度/指数值')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path_prefix}_{target_name}_predictions.png")
    plt.close()

def plot_anomalies(timestamps_or_indices, actual_values, anomaly_indices, target_name, save_path_prefix, title_suffix="异常点"):
    plt.figure(figsize=(15, 7))
    actual_flat = actual_values.flatten() 
    plt.plot(timestamps_or_indices, actual_flat, label=f'实际值 ({target_name})', alpha=0.7)
    valid_anomaly_indices = np.array([idx for idx in anomaly_indices if 0 <= idx < len(actual_flat)], dtype=int)
    if len(valid_anomaly_indices) > 0:
        if isinstance(timestamps_or_indices, (pd.DatetimeIndex, pd.RangeIndex)):
            plot_x_values = np.array(timestamps_or_indices)
        else: 
            plot_x_values = np.asarray(timestamps_or_indices)
        anomaly_x = plot_x_values[valid_anomaly_indices]
        anomaly_y = actual_flat[valid_anomaly_indices]
        plt.scatter(anomaly_x, anomaly_y, color='red', label='检测到的异常点 (Anomaly)', marker='o', s=50, zorder=5)
    plt.title(f'{target_name} - {title_suffix}')
    plt.xlabel('时间 / 序列索引') 
    plt.ylabel(f'{target_name} 浓度/指数值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    save_full_path = f"{save_path_prefix}_anomalies.png" 
    plt.savefig(save_full_path)
    plt.close()

def detect_anomalies_iqr_and_impute(df, column_names, factor=1.5, interpolation_method='time'):
    df_cleaned = df.copy() 
    print("开始对指定列进行基于IQR的异常值检测和插值填充...")
    for col_name in column_names:
        if col_name in df_cleaned.columns:
            if not pd.api.types.is_numeric_dtype(df_cleaned[col_name]):
                print(f"警告: 列 '{col_name}' 非数值类型，跳过其异常值处理流程。")
                continue
            original_nan_count = df_cleaned[col_name].isna().sum()
            Q1 = df_cleaned[col_name].quantile(0.25)
            Q3 = df_cleaned[col_name].quantile(0.75)
            IQR = Q3 - Q1 
            if pd.notna(IQR) and IQR > 1e-6: 
                lower_bound = Q1 - factor * IQR 
                upper_bound = Q3 + factor * IQR 
                outlier_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
                num_outliers = outlier_mask.sum()
                if num_outliers > 0: 
                    print(f"列 '{col_name}': 检测到 {num_outliers} 个基于IQR的统计学异常值。将其标记为NaN以便后续插值。")
                    df_cleaned.loc[outlier_mask, col_name] = np.nan 
            else:
                print(f"列 '{col_name}': IQR值为0或无效 ({IQR})，跳过基于IQR的异常值标记步骤。")
            if isinstance(df_cleaned.index, pd.DatetimeIndex): 
                try: 
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method=interpolation_method, limit_direction='both')
                except Exception as e: 
                    print(f"列 '{col_name}' 时间插值失败: {e}。尝试使用线性插值作为备选。")
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            else: 
                df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            if df_cleaned[col_name].isna().sum() > 0 : 
                median_val = df_cleaned[col_name].median() 
                fill_value = median_val if pd.notna(median_val) else 0 
                df_cleaned[col_name] = df_cleaned[col_name].fillna(fill_value)
                print(f"列 '{col_name}': 剩余的NaN值已使用中位数({median_val:.2f})或0 (若中位数无效，则为{fill_value:.2f})进行填充。")
        else: 
            print(f"警告: 列 '{col_name}' 在DataFrame中未找到，跳过其异常值处理。")
    print("基于IQR的异常值检测和插值填充流程完成。")
    return df_cleaned

class TimeSeriesDataset(TensorDataset):
    def __init__(self, X, y): 
        X_tensor = torch.from_numpy(X).float() # X已经是float32
        y_tensor = torch.from_numpy(y).float() # y已经是float32
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor)

class AQITransformer(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, num_target_features, norm_first=True):
        super(AQITransformer, self).__init__()
        self.d_model = d_model 
        self.horizon = horizon 
        self.num_target_features = num_target_features 
        self.input_embedding = nn.Linear(num_features, d_model) 
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
        self.output_layer = nn.Linear(d_model, horizon * num_target_features) 
    
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
        output_flat = self.output_layer(prediction_input) 
        output = output_flat.view(output_flat.size(0), self.horizon, self.num_target_features)
        return output

class ModelTrainer:
    def __init__(self, config):
        self.config = config 
        self.all_feature_columns_for_sequence = [] 
        self.feature_scaler = None 
        self.target_scalers = {}   

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True):
        print(f"开始从文件加载并执行核心数据预处理流程: {file_path}...")
        try: 
            df = pd.read_csv(file_path)
        except Exception: 
            try: 
                df = pd.read_excel(file_path)
            except Exception as e_excel: 
                print(f"错误: 读取CSV和Excel文件均失败。路径: '{file_path}'. 详细错误: {e_excel}")
                raise 
        if 'date' in df.columns and 'hour' in df.columns: 
            try:
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(int).astype(str).str.zfill(2), format='%Y%m%d%H')
                df = df.set_index('timestamp').drop(columns=['date', 'hour'], errors='ignore') 
            except Exception as e_dt:
                print(f"警告: 从'date'和'hour'列创建时间戳索引失败: {e_dt}。尝试其他方法...")
        elif 'Time' in df.columns: 
             try:
                df['timestamp'] = pd.to_datetime(df['Time'])
                df = df.set_index('timestamp').drop(columns=['Time'], errors='ignore')
             except Exception as e_t:
                print(f"警告: 从'Time'列创建时间戳索引失败: {e_t}。尝试其他方法...")
        else: 
            try: 
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception as e_first_col: 
                raise ValueError(f"错误: 无法自动从数据中解析并设置时间索引。请确保数据包含 ('date' 和 'hour') 列组合，或 'Time' 列，或第一列是可被pandas解析的时间格式。失败详情: {e_first_col}")
        for col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce') 
        feature_candidate_cols = [col for col in df.columns if col not in self.config['target_col_names']]
        for col in feature_candidate_cols: 
            if df[col].isnull().any(): 
                df[col] = df[col].ffill().bfill().fillna(0) 
        if fit_scalers: 
            original_len = len(df)
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
            if len(df) < original_len:
                print(f"信息: 由于目标列存在NaN值，已删除 {original_len - len(df)} 行数据。")
        if df.empty: 
            raise ValueError("错误: 数据在初步NaN处理（特别是目标列NaN行删除）后变为空。请检查原始数据质量。")
        if fit_scalers: 
            df = detect_anomalies_iqr_and_impute(df, self.config['target_col_names'])
            original_len_after_iqr = len(df)
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names']) 
            if len(df) < original_len_after_iqr:
                 print(f"信息: 在IQR异常处理和插值后，由于目标列仍存在NaN，又删除了 {original_len_after_iqr - len(df)} 行。")
        if df.empty: 
            raise ValueError("错误: 数据在IQR异常值处理流程后变为空。")
        df = calculate_aqi_from_pollutants(df.copy()) 
        if 'AQI_calculated' in df.columns and df['AQI_calculated'].isnull().any():
            df['AQI_calculated'] = df['AQI_calculated'].fillna(0)
        new_cyclical_features = pd.DataFrame(index=df.index) 
        if isinstance(df.index, pd.DatetimeIndex): 
            idx = df.index
            new_cyclical_features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['month_sin'] = np.sin(2 * np.pi * idx.month / 12.0) 
            new_cyclical_features['month_cos'] = np.cos(2 * np.pi * idx.month / 12.0)
            df = pd.concat([df, new_cyclical_features], axis=1) 
        else:
            print("警告: DataFrame索引不是DatetimeIndex，无法创建周期性时间特征。")
        num_lags_to_create = max(1, self.config['look_back'] // 4) 
        lag_features_to_concat = [] 
        lag_cols_created_names = [] 
        cols_for_lags = self.config['target_col_names'] + \
                        [col for col in df.columns if (col.endswith('_24h') or col.endswith('_8h')) and col not in self.config['target_col_names']]
        cols_for_lags = sorted(list(set(cols_for_lags))) 
        for col_to_lag in cols_for_lags: 
            if col_to_lag in df.columns: 
                for lag in range(1, num_lags_to_create + 1): 
                    lag_col_name = f"{col_to_lag}_lag_{lag}" 
                    lag_features_to_concat.append(df[col_to_lag].shift(lag).rename(lag_col_name))
                    lag_cols_created_names.append(lag_col_name)
        if lag_features_to_concat: 
            df = pd.concat([df] + lag_features_to_concat, axis=1) 
            original_len_before_lag_dropna = len(df)
            df = df.dropna(subset=lag_cols_created_names, how='any') 
            if len(df) < original_len_before_lag_dropna:
                print(f"信息: 由于创建滞后特征引入NaN，已删除 {original_len_before_lag_dropna - len(df)} 行数据。")
        if df.empty: 
            raise ValueError("错误: 数据在创建滞后特征并处理相关NaN后变为空。")
        self.all_feature_columns_for_sequence = [col for col in df.columns if col not in self.config['target_col_names']]
        if 'AQI_calculated' in df.columns and 'AQI_calculated' not in self.config['target_col_names']:
            if 'AQI_calculated' not in self.all_feature_columns_for_sequence: 
                 self.all_feature_columns_for_sequence.append('AQI_calculated')
        elif 'AQI_calculated' in self.all_feature_columns_for_sequence and 'AQI_calculated' in self.config['target_col_names']:
             print("配置警告: 'AQI_calculated' 列同时被识别为模型输入特征和预测目标。这可能不符合预期，并可能导致数据泄露。请检查target_col_names配置。")
        self.all_feature_columns_for_sequence = sorted(list(set(self.all_feature_columns_for_sequence))) 
        missing_targets_for_scaling = [tc for tc in self.config['target_col_names'] if tc not in df.columns]
        if missing_targets_for_scaling: 
            raise ValueError(f"错误: 一个或多个目标列在准备进行数据缩放前未在DataFrame中找到: {missing_targets_for_scaling}.")
        if fit_scalers:
            print("开始拟合和保存数据缩放器...")
            self.feature_scaler = StandardScaler() 
            current_features_to_scale = [f_col for f_col in self.all_feature_columns_for_sequence if f_col in df.columns] 
            for f_col in current_features_to_scale: 
                if not pd.api.types.is_numeric_dtype(df[f_col]):
                    df[f_col] = pd.to_numeric(df[f_col], errors='coerce') 
                    if df[f_col].isnull().any(): 
                        df[f_col] = df[f_col].fillna(0) 
                    if not pd.api.types.is_numeric_dtype(df[f_col]): 
                         offending_values = [item for item in df[f_col].unique() if not isinstance(item, (int, float, np.number))]
                         raise ValueError(f"错误: 特征列 '{f_col}' 最终无法转换为数值类型以进行缩放。问题值示例: {offending_values[:5]}")
            if not current_features_to_scale: 
                raise ValueError("错误: 没有有效的特征列可用于拟合特征缩放器。")
            df[current_features_to_scale] = self.feature_scaler.fit_transform(df[current_features_to_scale])
            joblib.dump(self.feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))
            self.target_scalers = {} 
            for col_name in self.config['target_col_names']:
                if not pd.api.types.is_numeric_dtype(df[col_name]): 
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0) 
                scaler = StandardScaler() 
                df[[col_name]] = scaler.fit_transform(df[[col_name]]) 
                self.target_scalers[col_name] = scaler 
            joblib.dump(self.target_scalers, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALERS_SAVE_NAME))
            print("特征缩放器和各目标变量的缩放器均已成功拟合和保存。")
        return df

    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, trial=None):
        best_val_loss = float('inf') 
        epochs_no_improve = 0        
        best_model_state = None      
        train_losses_epoch = [] 
        val_losses_epoch = []   
        primary_target_idx = self.config['target_col_names'].index(self.config['primary_target_col_name'])
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
            epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
            train_losses_epoch.append(epoch_train_loss)
            model.eval() 
            running_val_loss = 0.0 
            running_primary_target_val_loss = 0.0 
            if len(val_loader.dataset) > 0: 
                with torch.no_grad(): 
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        running_val_loss += loss.item() * X_batch.size(0)
                        primary_target_loss = criterion(outputs[:, :, primary_target_idx], y_batch[:, :, primary_target_idx])
                        running_primary_target_val_loss += primary_target_loss.item() * X_batch.size(0)
                epoch_val_loss = running_val_loss / len(val_loader.dataset) 
                epoch_primary_target_val_loss = running_primary_target_val_loss / len(val_loader.dataset) 
            else: 
                epoch_val_loss = float('inf')
                epoch_primary_target_val_loss = float('inf')
            val_losses_epoch.append(epoch_val_loss) 
            current_lr = optimizer.param_groups[0]['lr'] 
            print(f"轮次 [{epoch+1}/{epochs}], 学习率: {current_lr:.7f}, "
                  f"训练损失: {epoch_train_loss:.6f}, "
                  f"验证损失 (综合): {epoch_val_loss:.6f}, "
                  f"验证损失 ({self.config['primary_target_col_name']}): {epoch_primary_target_val_loss:.6f}")
            if scheduler: 
                scheduler.step(epoch_primary_target_val_loss) 
            if trial: 
                trial.report(epoch_primary_target_val_loss, epoch) 
                if trial.should_prune(): 
                    print("Optuna试验被剪枝 (Optuna trial pruned).")
                    raise optuna.exceptions.TrialPruned() 
            if epoch_primary_target_val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = epoch_primary_target_val_loss 
                epochs_no_improve = 0 
                best_model_state = copy.deepcopy(model.state_dict()) 
                if trial is None: 
                    torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
                    print(f"验证损失 ({self.config['primary_target_col_name']}) 获得改善。模型状态已于轮次 {epoch+1} 保存。")
            else: 
                epochs_no_improve += 1
            if epochs_no_improve >= self.config['early_stopping_patience'] and len(val_loader.dataset) > 0 : 
                print(f"早停机制在轮次 {epoch+1} 被触发 (基于 {self.config['primary_target_col_name']} 的验证损失连续 {self.config['early_stopping_patience']} 轮未改善)。")
                if best_model_state: 
                    model.load_state_dict(best_model_state) 
                break 
        if best_model_state and trial is None: 
             model.load_state_dict(best_model_state) 
             torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
             print(f"训练结束。最终最佳模型状态已保存至 {MODEL_STATE_SAVE_NAME}")
        return model, train_losses_epoch, val_losses_epoch, best_val_loss

    def _objective_optuna(self, trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features):
        lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True) 
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512]) 
        possible_num_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and d_model >= h]
        if not possible_num_heads: 
            raise optuna.exceptions.TrialPruned("对于当前d_model，没有有效的注意力头数选项。")
        num_heads = trial.suggest_categorical('num_heads', possible_num_heads)
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8) 
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 6) 
        dim_feedforward = d_model * dim_feedforward_factor
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4) 
        norm_first = trial.suggest_categorical('norm_first', [True, False])
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.7) 
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)  
        model = AQITransformer(
            num_features=num_input_features, 
            d_model=d_model, nhead=num_heads, 
            num_encoder_layers=num_encoder_layers, 
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate, 
            norm_first=norm_first, 
            horizon=self.config['horizon'], 
            num_target_features=len(self.config['target_col_names'])
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False) 
        criterion = nn.MSELoss() 
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True) 
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        print(f"\nOptuna试验 {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, "
              f"ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.3f}, norm_first={norm_first}, "
              f"wd={weight_decay:.7f}, sch_factor={scheduler_factor:.2f}, sch_patience={scheduler_patience}")
        _, _, _, best_val_loss_trial = self._train_model_core(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            epochs=self.config['optuna_epochs'], 
            trial=trial 
        )
        return best_val_loss_trial 

    def run_training_pipeline(self):
        print("--- 开始执行模型训练全流程 ---")
        df_processed = self._load_and_preprocess_data_core(self.config['file_path'], fit_scalers=True)
        num_input_features_for_model = len(self.all_feature_columns_for_sequence) 
        if num_input_features_for_model == 0:
            print("错误: 预处理后没有有效的输入特征列。终止训练。")
            return

        # 添加诊断打印
        print(f"DEBUG: 在 run_training_pipeline 中, 调用 create_sequences 之前:")
        print(f"DEBUG: self.config['look_back'] = {self.config['look_back']}")
        print(f"DEBUG: self.config['horizon'] = {self.config['horizon']}")
        print(f"DEBUG: 特征数量 (len(self.all_feature_columns_for_sequence)) = {len(self.all_feature_columns_for_sequence)}")

        X_initial, y_initial = create_sequences(
            df_processed, 
            self.config['look_back'], 
            self.config['horizon'], 
            self.config['target_col_names'], 
            self.all_feature_columns_for_sequence 
        )
        if X_initial.size == 0 or y_initial.size == 0: 
            print("错误: 创建输入/输出序列后数据为空。可能是原始数据过短或预处理问题。终止训练。")
            return
        total_samples = X_initial.shape[0]
        train_idx_end = int(total_samples * 0.7)
        val_idx_end = int(total_samples * 0.85) 
        X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]
        X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
        X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:]
        print(f"数据集划分完毕: 训练集样本数={X_train_np.shape[0]}, 验证集样本数={X_val_np.shape[0]}, 测试集样本数={X_test_np.shape[0]}")
        if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0: 
            print("错误: 训练集或验证集在划分后为空。无法继续训练。请检查数据量。")
            return
        print("\n--- 开始Optuna超参数优化 ---")
        study = optuna.create_study(
            direction='minimize', 
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, 
                max_resource=self.config['optuna_epochs'], 
                reduction_factor=3
            ), 
            sampler=optuna.samplers.TPESampler(seed=42) 
        ) 
        study.optimize(
            lambda trial: self._objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features_for_model), 
            n_trials=self.config['n_optuna_trials'], 
            timeout=3600*6 
        ) 
        best_hyperparams = study.best_params 
        print(f"Optuna超参数优化完成。")
        print(f"最佳试验的验证损失 ({self.config['primary_target_col_name']}): {study.best_value:.6f}")
        print(f"找到的最佳超参数组合: {best_hyperparams}")
        print("\n--- 使用最佳超参数训练最终模型 ---")
        final_model_arch_params = {
            'd_model': best_hyperparams['d_model'], 
            'nhead': best_hyperparams['num_heads'], 
            'num_encoder_layers': best_hyperparams['num_encoder_layers'],
            'dim_feedforward': best_hyperparams['d_model'] * best_hyperparams['dim_feedforward_factor'], 
            'dropout': best_hyperparams['dropout_rate'], 
            'norm_first': best_hyperparams['norm_first']
        }
        final_model = AQITransformer(
            num_features=num_input_features_for_model, 
            **final_model_arch_params, 
            horizon=self.config['horizon'], 
            num_target_features=len(self.config['target_col_names'])
        ).to(DEVICE)
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        final_optimizer = torch.optim.AdamW(
            final_model.parameters(), 
            lr=best_hyperparams['learning_rate'], 
            weight_decay=best_hyperparams.get('weight_decay', 0.0) 
        )
        final_scheduler = ReduceLROnPlateau(
            final_optimizer, mode='min', 
            factor=best_hyperparams.get('scheduler_factor', 0.5), 
            patience=best_hyperparams.get('scheduler_patience', 7),
            verbose=False
        ) 
        criterion = nn.MSELoss() 
        final_model, train_losses, val_losses, _ = self._train_model_core(
            final_model, final_train_loader, final_val_loader, criterion, final_optimizer, final_scheduler, 
            epochs=self.config['full_train_epochs'] 
        )
        plot_training_loss(train_losses, val_losses, 
                           os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"), 
                           title_prefix="最终模型")
        print(f"最终模型训练完成。最佳模型状态已保存。")
        model_config_to_save = {
            'model_architecture': final_model_arch_params, 
            'look_back': self.config['look_back'], 
            'horizon': self.config['horizon'],
            'target_col_names': self.config['target_col_names'], 
            'primary_target_col_name': self.config['primary_target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence, 
            'num_input_features_for_model': num_input_features_for_model,
            'num_target_features': len(self.config['target_col_names']), 
            'optuna_best_params': best_hyperparams 
        } 
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w', encoding='utf-8') as f:
            json.dump(model_config_to_save, f, indent=4, ensure_ascii=False) 
        print(f"模型配置信息已保存至 {MODEL_CONFIG_SAVE_NAME}。")
        if X_test_np.shape[0] > 0: 
            self.evaluate_trained_model(final_model, X_test_np, y_test_np, criterion)
        else:
            print("测试集为空，跳过最终模型评估步骤。")

    def evaluate_trained_model(self, model, X_test_np, y_test_np, criterion):
        print("\n--- 开始在测试集上评估最终模型性能 ---")
        test_dataset = TimeSeriesDataset(X_test_np, y_test_np)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        model.eval() 
        all_preds_scaled_list = []   
        all_targets_scaled_list = [] 
        with torch.no_grad(): 
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                outputs_scaled = model(X_batch) 
                all_preds_scaled_list.append(outputs_scaled.cpu().numpy())
                all_targets_scaled_list.append(y_batch.numpy()) 
        if not all_preds_scaled_list: 
            print("错误: 在测试集上未能生成任何预测。评估中止。")
            return
        preds_scaled_np = np.concatenate(all_preds_scaled_list, axis=0)
        targets_scaled_np = np.concatenate(all_targets_scaled_list, axis=0)
        actual_orig_all_targets = np.zeros_like(targets_scaled_np) 
        predicted_orig_all_targets = np.zeros_like(preds_scaled_np) 
        for i, col_name in enumerate(self.config['target_col_names']):
            scaler = self.target_scalers[col_name] 
            actual_orig_all_targets[:, :, i] = scaler.inverse_transform(targets_scaled_np[:, :, i])
            predicted_orig_all_targets[:, :, i] = scaler.inverse_transform(preds_scaled_np[:, :, i])
        print("\n各目标污染物在测试集上的评估指标 (原始尺度):")
        for i, col_name in enumerate(self.config['target_col_names']):
            actual_col_flat = actual_orig_all_targets[:, :, i].flatten()
            predicted_col_flat = predicted_orig_all_targets[:, :, i].flatten()
            mae = mean_absolute_error(actual_col_flat, predicted_col_flat)
            rmse = np.sqrt(mean_squared_error(actual_col_flat, predicted_col_flat))
            r2 = r2_score(actual_col_flat, predicted_col_flat)
            print(f"  {col_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            plot_save_prefix = os.path.join(self.config['model_artifacts_dir'], f"final_model_evaluation_test_set") 
            plot_predictions_vs_actual(
                actual_orig_all_targets[:, :, i], 
                predicted_orig_all_targets[:, :, i], 
                col_name, 
                plot_save_prefix, 
                title_suffix="测试集评估：实际值 vs. 预测值"
            )
        print(f"\n测试集评估图表已保存至目录: {self.config['model_artifacts_dir']}")
        print("\n提示: AQI是基于多种污染物（PM2.5, PM10, SO2, NO2, O3, CO）浓度通过特定算法计算得出的综合性空气质量评价指数。")

class ModelPredictor: 
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.feature_scaler = None
        self.target_scalers = {}
        self.model_config = None
        self._load_artifacts() 

    def _load_artifacts(self):
        print(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (ModelPredictor)...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME) 
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME) 
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing_files = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing_files}。请确保模型已正确训练并保存。")
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.model_config = json.load(f)
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'], 
            **self.model_config['model_architecture'], 
            horizon=self.model_config['horizon'], 
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval() 
        print("模型、配置及缩放器已成功加载 (ModelPredictor)。")

    def _preprocess_input_for_prediction(self, df_raw):
        print("开始为预测任务预处理输入数据 (ModelPredictor)...")
        df_processed = df_raw.copy() 
        if not isinstance(df_processed.index, pd.DatetimeIndex): 
            if 'Time' in df_processed.columns: 
                df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
                df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                 df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                 df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                 df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: 
                try:
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                        df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0])
                        df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: 
                    raise ValueError(f"错误: 无法为预测数据自动设置时间索引: {e}。请检查输入数据格式。")
        for col in df_processed.columns: 
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 
        feature_cols_in_df = [col for col in df_processed.columns if col not in self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)]
        for col in feature_cols_in_df: 
            if df_processed[col].isnull().any(): 
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        if 'AQI_calculated' in self.model_config['all_feature_columns_for_sequence']:
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi: 
                df_processed = calculate_aqi_from_pollutants(df_processed.copy()) 
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                    df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0) 
            else: 
                print("警告: 预测数据中缺少计算'AQI_calculated'所需的全部污染物列。将 'AQI_calculated' 特征填充为0。")
                df_processed['AQI_calculated'] = 0 
        new_cyclical_features_pred = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex): 
            idx = df_processed.index
            new_cyclical_features_pred['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0)
            new_cyclical_features_pred['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features_pred['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features_pred['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features_pred['month_sin'] = np.sin(2*np.pi*idx.month/12.0)
            new_cyclical_features_pred['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
            df_processed = pd.concat([df_processed, new_cyclical_features_pred], axis=1)
        else:
            print("警告: 预测数据的索引不是DatetimeIndex，无法创建周期性时间特征。如果模型依赖这些特征，预测可能不准确。")
        lag_features_to_recreate = [f for f in self.model_config['all_feature_columns_for_sequence'] if "_lag_" in f] 
        lag_series_list_pred = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_")
            if len(original_col_name_parts) == 2: 
                original_col = original_col_name_parts[0]
                lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns: 
                    lag_series_list_pred.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: 
                    print(f"警告: 创建滞后特征'{lag_col_name}'所需的原始列'{original_col}'在预测数据中未找到。该滞后特征将填充为0。")
                    lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: 
                print(f"警告: 滞后特征名 '{lag_col_name}' 格式无法解析。该特征将填充为0。")
                lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        if lag_series_list_pred:
             df_processed = pd.concat([df_processed] + lag_series_list_pred, axis=1)
        expected_features_from_config = self.model_config['all_feature_columns_for_sequence'] 
        df_for_scaling = pd.DataFrame(index=df_processed.index) 
        for f_col in expected_features_from_config:
            if f_col not in df_processed.columns: 
                print(f"警告: 模型期望的特征列 '{f_col}' 在预测数据预处理后仍缺失。将创建为全0列。")
                df_for_scaling[f_col] = 0 
            else:
                if df_processed[f_col].isnull().any():
                    df_processed[f_col] = df_processed[f_col].ffill().bfill().fillna(0)
                df_for_scaling[f_col] = df_processed[f_col] 
        for f_col in expected_features_from_config: 
            df_for_scaling[f_col] = pd.to_numeric(df_for_scaling[f_col], errors='coerce').fillna(0) 
            if not pd.api.types.is_numeric_dtype(df_for_scaling[f_col]):
                try: 
                    df_for_scaling[f_col] = df_for_scaling[f_col].astype(float)
                except ValueError as e_astype:
                    offending_values = [item for item in df_for_scaling[f_col].unique() if not isinstance(item, (int, float, np.number))]
                    raise ValueError(f"错误: 无法将预测数据中的特征列 '{f_col}' 转换为浮点数以进行缩放。问题值示例: {offending_values[:5]}. 原错误: {e_astype}")
        df_for_scaling_transformed = self.feature_scaler.transform(df_for_scaling[expected_features_from_config]) 
        df_scaled_features = pd.DataFrame(df_for_scaling_transformed, columns=expected_features_from_config, index=df_for_scaling.index)
        print("预测输入数据预处理完成。")
        return df_scaled_features

    def predict(self, input_data_path_or_df):
        if isinstance(input_data_path_or_df, str):
            try: 
                df_raw = pd.read_csv(input_data_path_or_df)
            except Exception: 
                try:
                    df_raw = pd.read_excel(input_data_path_or_df)
                except FileNotFoundError:
                    raise FileNotFoundError(f"错误: 预测数据文件 '{input_data_path_or_df}' 未找到。")
                except Exception as e_excel:
                    raise ValueError(f"错误: 读取Excel预测数据文件 '{input_data_path_or_df}' 失败: {e_excel}")
        elif isinstance(input_data_path_or_df, pd.DataFrame):
            df_raw = input_data_path_or_df.copy() 
        else:
            raise ValueError("错误: 输入数据必须是CSV/Excel文件路径或Pandas DataFrame。")
        last_known_timestamp = None 
        if isinstance(df_raw.index, pd.DatetimeIndex) and not df_raw.empty: 
            last_known_timestamp = df_raw.index[-1]
        elif not df_raw.empty: 
            time_cols_to_check = ['Time', 'timestamp', 'Datetime'] 
            for tc in time_cols_to_check:
                if tc in df_raw.columns: 
                    try: last_known_timestamp = pd.to_datetime(df_raw[tc].iloc[-1]); break
                    except: pass 
            if last_known_timestamp is None and 'date' in df_raw.columns and 'hour' in df_raw.columns: 
                try:
                    date_val = str(df_raw['date'].iloc[-1])
                    hour_val = str(int(df_raw['hour'].iloc[-1])).zfill(2) 
                    last_dt_str = date_val + hour_val
                    last_known_timestamp = pd.to_datetime(last_dt_str, format='%Y%m%d%H')
                except: pass
            elif last_known_timestamp is None: 
                 try:
                     if not pd.api.types.is_numeric_dtype(df_raw.iloc[:,0]): 
                         last_known_timestamp = pd.to_datetime(df_raw.iloc[-1, 0])
                 except: pass 
        if last_known_timestamp is None: 
            print("警告: 无法从输入数据中可靠地确定最后一个时间戳。预测结果的时间标签可能不准确。")
        df_processed_features = self._preprocess_input_for_prediction(df_raw.copy()) 
        if len(df_processed_features) < self.model_config['look_back']:
            raise ValueError(f"错误: 经过预处理后的预测输入数据长度为 {len(df_processed_features)}，"
                             f"不足以满足模型所需的回溯窗口长度 {self.model_config['look_back']}。")
        model_input_feature_names = self.model_config['all_feature_columns_for_sequence']
        available_model_features = [col for col in model_input_feature_names if col in df_processed_features.columns]
        if len(available_model_features) != len(model_input_feature_names):
            missing_for_pred = set(model_input_feature_names) - set(available_model_features)
            print(f"警告: 预处理后，部分模型期望的特征列在最终输入数据中缺失: {missing_for_pred}。这些列可能已被填充为0，但请核实。")
        last_input_sequence_df = df_processed_features[available_model_features].iloc[-self.model_config['look_back']:]
        X_pred_np = np.array([last_input_sequence_df.values]) 
        if X_pred_np.size == 0: 
            print("错误: 无法创建用于预测的输入序列 (X_pred_np为空)。")
            return None, last_known_timestamp 
        X_pred_torch = torch.from_numpy(X_pred_np).float().to(DEVICE) 
        with torch.no_grad(): 
            predictions_scaled = self.model(X_pred_torch) 
        predictions_scaled_np_slice = predictions_scaled.cpu().numpy()[0, :, :] 
        predictions_original_all_targets = np.zeros_like(predictions_scaled_np_slice) 
        for i, col_name in enumerate(self.model_config['target_col_names']):
            scaler = self.target_scalers[col_name] 
            pred_col_scaled_reshaped = predictions_scaled_np_slice[:, i].reshape(-1, 1) 
            predictions_original_all_targets[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten() 
        print(f"已成功生成 {self.model_config['horizon']} 个时间步的预测。")
        return predictions_original_all_targets, last_known_timestamp

class AQISystem:
    def __init__(self, artifacts_dir=MODEL_ARTIFACTS_DIR, config_overrides=None):
        self.artifacts_dir = artifacts_dir
        self.config = self._load_default_config() 
        if config_overrides: 
            self.config.update(config_overrides)
        self.config['model_artifacts_dir'] = self.artifacts_dir 
        os.makedirs(self.artifacts_dir, exist_ok=True) 
        set_seed() 
        self.trainer = None 
        self.predictor_instance = None 
        self.model_config = None
        self.feature_scaler = None
        self.target_scalers = None
        self.model = None
        self.all_feature_columns_for_sequence = None 

    def _load_default_config(self):
        return {
            'file_path': DEFAULT_FILE_PATH, 
            'look_back': DEFAULT_LOOK_BACK, 
            'horizon': DEFAULT_HORIZON,
            'target_col_names': DEFAULT_TARGET_COL_NAMES, 
            'primary_target_col_name': DEFAULT_PRIMARY_TARGET_COL_NAME,
            'batch_size': DEFAULT_BATCH_SIZE, 
            'model_artifacts_dir': self.artifacts_dir, 
            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS, 
            'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS, 
            'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA, 
            'anomaly_threshold_factor': DEFAULT_ANOMALY_THRESHOLD_FACTOR 
        }

    def _ensure_model_loaded_for_use(self):
        if self.model is not None and self.model_config is not None and \
           self.feature_scaler is not None and self.target_scalers is not None and \
           self.all_feature_columns_for_sequence is not None:
            return 
        print(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (AQISystem集中管理)...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing}。请确保模型已成功训练并保存，或指定的目录正确。")
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.model_config = json.load(f)
        self.config['look_back'] = self.model_config.get('look_back', self.config['look_back'])
        self.config['horizon'] = self.model_config.get('horizon', self.config['horizon'])
        self.config['target_col_names'] = self.model_config.get('target_col_names', self.config['target_col_names'])
        self.config['primary_target_col_name'] = self.model_config.get('primary_target_col_name', self.config['primary_target_col_name'])
        self.all_feature_columns_for_sequence = self.model_config.get('all_feature_columns_for_sequence')
        if self.all_feature_columns_for_sequence is None:
            raise ValueError("错误: 加载的模型配置 (model_config.json) 中未找到 'all_feature_columns_for_sequence'。此信息对于数据预处理至关重要。")
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'],
            **self.model_config['model_architecture'], 
            horizon=self.model_config['horizon'],
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)) 
        self.model.eval() 
        print("模型、配置及缩放器已成功加载并准备就绪 (AQISystem集中管理)。")

    def train_new_model(self, train_data_path):
        self.config['file_path'] = train_data_path 
        self.trainer = ModelTrainer(self.config) 
        try:
            self.trainer.run_training_pipeline() 
            print("模型训练完成。尝试加载新训练的模型组件到当前AQISystem实例...")
            self._ensure_model_loaded_for_use() 
        except Exception as e:
            print(f"模型训练过程中发生严重错误: {e}")
            import traceback; traceback.print_exc() 

    def predict_with_existing_model(self, input_data_path_or_df):
        try:
            self._ensure_model_loaded_for_use() 
            if self.predictor_instance is None or self.predictor_instance.artifacts_dir != self.artifacts_dir:
                 self.predictor_instance = ModelPredictor(self.artifacts_dir)
            predicted_values, last_timestamp = self.predictor_instance.predict(input_data_path_or_df)
            if predicted_values is not None: 
                horizon_from_pred_config = self.predictor_instance.model_config.get('horizon', self.config['horizon'])
                target_names_from_pred_config = self.predictor_instance.model_config.get('target_col_names', self.config['target_col_names'])
                print(f"\n模型成功生成了未来 {horizon_from_pred_config} 小时的各项指标预测值 (原始尺度):")
                for h in range(min(5, horizon_from_pred_config)): 
                    hour_str = f"  未来第 {h+1} 小时: "
                    for t_idx, t_name in enumerate(target_names_from_pred_config): 
                        val_to_print = predicted_values[h, t_idx]
                        if t_name == 'CO': 
                            hour_str += f"{t_name}={val_to_print:.2f} "
                        else: 
                            hour_str += f"{t_name}={np.round(val_to_print).astype(int)} "
                    print(hour_str)
                if last_timestamp is not None: 
                    save_pred = input("是否将详细预测结果保存到CSV文件? (y/n): ").strip().lower() 
                    if save_pred.startswith('y'): 
                        pred_save_path = os.path.join(self.artifacts_dir, "predictions_output_adv.csv") 
                        future_timestamps = pd.date_range(
                            start=last_timestamp + pd.Timedelta(hours=1), 
                            periods=horizon_from_pred_config, 
                            freq='H' 
                        )
                        output_data = {'date': future_timestamps.strftime('%Y%m%d'), 'hour': future_timestamps.hour}
                        for t_idx, t_name in enumerate(target_names_from_pred_config): 
                            val_col = predicted_values[:, t_idx] 
                            if t_name == 'CO': 
                                output_data[t_name] = np.round(val_col, 2)
                            else: 
                                output_data[t_name] = np.round(val_col).astype(int)
                        requested_output_columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
                        final_output_df_cols = {}
                        for col in requested_output_columns:
                            final_output_df_cols[col] = output_data.get(col, [np.nan] * horizon_from_pred_config)
                        output_df = pd.DataFrame(final_output_df_cols)[requested_output_columns] 
                        output_df.to_csv(pred_save_path, index=False, encoding='utf-8-sig') 
                        print(f"预测结果已保存至: {pred_save_path}")
                        predicted_cols_in_output = [col for col in target_names_from_pred_config if col in requested_output_columns]
                        nan_cols_in_output = [col for col in requested_output_columns if col not in ['date', 'hour'] and col not in predicted_cols_in_output]
                        print(f"\n提示: CSV文件中实际包含预测数据的列为: {', '.join(predicted_cols_in_output)}。")
                        if nan_cols_in_output: 
                            print(f"以下在标准输出列中但未被当前模型预测的列，在CSV中已填充为NaN: {', '.join(nan_cols_in_output)}。")
                    else: 
                        print("预测结果未保存。")
                else: 
                    print("由于无法从输入数据中确定最后一个有效时间戳，预测结果未关联具体日期，也无法保存为带时间戳的CSV文件。")
                    print("原始预测值数组 (形状: horizon, num_targets):")
                    print(np.round(predicted_values,2)) 
            elif predicted_values is None: 
                print("模型未能成功生成预测。请检查日志或输入数据。")
        except FileNotFoundError as e:
             print(f"文件未找到错误 (预测流程): {e}。请检查模型工件目录和输入文件路径。")
        except ValueError as e:
             print(f"值错误 (预测流程): {e}")
        except Exception as e:
             print(f"预测过程中发生未知错误: {e}")
             import traceback; traceback.print_exc()

    def _preprocess_input_for_anomaly(self, df_raw):
        print("开始为异常检测任务预处理输入数据 (AQISystem)...")
        df_processed = df_raw.copy() 
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            if 'Time' in df_processed.columns:
                df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
                df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: 
                try: 
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                         df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0])
                         df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: 
                    raise ValueError(f"错误: 无法为异常检测数据自动设置时间索引: {e}。")
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        feature_cols_in_df_anomaly = [col for col in df_processed.columns if col not in model_target_cols]
        for col in feature_cols_in_df_anomaly: 
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        for tc in model_target_cols: 
            if tc in df_processed.columns:
                if df_processed[tc].isnull().any():
                    print(f"警告: 目标列 '{tc}' 在用于异常检测的数据中包含NaN值。这些NaN值将被填充为0，可能影响异常检测的准确性。")
                    df_processed[tc] = df_processed[tc].fillna(0)
            else: 
                print(f"严重警告: 模型训练时定义的目标列 '{tc}' 在提供的异常检测数据中完全缺失。将为此列创建全0数据，但这很可能导致无效的异常检测结果。")
                df_processed[tc] = 0 
        if 'AQI_calculated' in self.all_feature_columns_for_sequence: 
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi:
                df_processed = calculate_aqi_from_pollutants(df_processed.copy()) 
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                     df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            else:
                print("警告: 异常检测数据缺少计算'AQI_calculated'所需的全部或部分污染物列。'AQI_calculated'特征将填充为0。")
                df_processed['AQI_calculated'] = 0
        new_cyclical_features = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            idx = df_processed.index
            new_cyclical_features['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
            df_processed = pd.concat([df_processed, new_cyclical_features], axis=1)
        else:
            if any(cyc_feat in self.all_feature_columns_for_sequence for cyc_feat in ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']):
                 print("警告: 异常检测数据的索引非DatetimeIndex，无法创建周期性时间特征。若模型依赖这些特征，结果可能不准确。")
        lag_features_to_recreate = [f for f in self.all_feature_columns_for_sequence if "_lag_" in f]
        lag_series_list = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_")
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]; lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns:
                    lag_series_list.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: 
                    print(f"警告: 创建滞后特征'{lag_col_name}'所需的原始列'{original_col}'在异常检测数据中未找到。该滞后特征填充为0。")
                    lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: 
                print(f"警告: 滞后特征名 '{lag_col_name}' 格式无法解析。该特征填充为0。")
                lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        if lag_series_list: 
            df_processed = pd.concat([df_processed] + lag_series_list, axis=1)
        for col in df_processed.columns: 
            if df_processed[col].isnull().any(): 
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        df_features_for_scaling = pd.DataFrame(index=df_processed.index) 
        for f_col in self.all_feature_columns_for_sequence: 
            if f_col not in df_processed.columns:
                print(f"警告: 模型期望的特征列 '{f_col}' 在异常检测数据特征工程后仍缺失。将创建为全0列用于缩放。")
                df_features_for_scaling[f_col] = 0 
            else:
                df_features_for_scaling[f_col] = df_processed[f_col]
        for f_col in self.all_feature_columns_for_sequence: 
             df_features_for_scaling[f_col] = pd.to_numeric(df_features_for_scaling[f_col], errors='coerce').fillna(0)
             if not pd.api.types.is_numeric_dtype(df_features_for_scaling[f_col]):
                try: 
                    df_features_for_scaling[f_col] = df_features_for_scaling[f_col].astype(float)
                except ValueError: 
                    raise ValueError(f"错误: 无法将异常检测数据中的特征列 '{f_col}' 转换为浮点数以进行缩放。")
        df_scaled_features_part = self.feature_scaler.transform(df_features_for_scaling[self.all_feature_columns_for_sequence])
        df_scaled_features_df = pd.DataFrame(df_scaled_features_part, columns=self.all_feature_columns_for_sequence, index=df_processed.index)
        df_final_for_sequences = df_scaled_features_df.copy()
        for tc in model_target_cols: 
            if tc in df_processed.columns: 
                df_final_for_sequences[tc] = df_processed[tc] 
            else: 
                 print(f"严重内部错误: 目标列 '{tc}' 在合并阶段未找到于df_processed。将填充为0。")
                 df_final_for_sequences[tc] = 0 
        cols_for_sequence_creation = self.all_feature_columns_for_sequence + model_target_cols
        for col in cols_for_sequence_creation:
            if col in df_final_for_sequences.columns and df_final_for_sequences[col].isnull().any():
                print(f"警告: 列 '{col}' 在准备创建序列用于异常检测前仍包含NaN值。将用0填充。")
                df_final_for_sequences[col] = df_final_for_sequences[col].fillna(0)
        if df_final_for_sequences.empty:
            raise ValueError("错误: 数据在为异常检测任务预处理完毕后变为空。请检查原始数据或预处理步骤。")
        print("异常检测输入数据预处理完成。")
        return df_final_for_sequences

    def detect_anomalies(self, data_path_or_df, threshold_factor=None):
        self._ensure_model_loaded_for_use() 
        current_threshold_factor = threshold_factor if threshold_factor is not None \
                                   else self.config.get('anomaly_threshold_factor', DEFAULT_ANOMALY_THRESHOLD_FACTOR)
        print(f"开始执行异常数据检测流程，使用的阈值因子为: {current_threshold_factor} ...")
        if isinstance(data_path_or_df, str):
            try: 
                df_raw_full = pd.read_csv(data_path_or_df)
            except Exception: 
                try:
                    df_raw_full = pd.read_excel(data_path_or_df)
                except FileNotFoundError:
                     raise FileNotFoundError(f"错误: 异常检测数据文件 '{data_path_or_df}' 未找到。")
                except Exception as e_excel:
                     raise ValueError(f"错误: 读取Excel异常检测数据文件 '{data_path_or_df}' 失败: {e_excel}")
        elif isinstance(data_path_or_df, pd.DataFrame):
            df_raw_full = data_path_or_df.copy()
        else:
            raise ValueError("错误: 异常检测的输入数据必须是CSV/Excel文件路径或Pandas DataFrame。")
        df_for_sequences = self._preprocess_input_for_anomaly(df_raw_full.copy())
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        missing_targets_in_processed_df = [tc for tc in model_target_cols 
                                           if tc not in df_for_sequences.columns or df_for_sequences[tc].isnull().all()]
        if missing_targets_in_processed_df:
            raise ValueError(f"错误: 异常检测数据在预处理后，缺少一个或多个有效的实际目标观测列: {missing_targets_in_processed_df}。无法进行异常比较。")
        X_anomaly_np, y_actual_anomaly_np = create_sequences(
            df_for_sequences, 
            look_back=self.model_config['look_back'], 
            horizon=1, 
            target_col_names=model_target_cols, 
            feature_cols=self.all_feature_columns_for_sequence, 
            is_predict=False
        )
        if X_anomaly_np.size == 0: 
            print("警告: 无法为异常检测创建有效的输入/输出序列。可能是数据量过少或预处理后数据不满足要求。异常检测中止。")
            return {} 
        self.model.eval() 
        all_predictions_scaled_list = []
        anomaly_dataset = TensorDataset(torch.from_numpy(X_anomaly_np).float())
        batch_size_for_anomaly = self.config.get('batch_size', DEFAULT_BATCH_SIZE) 
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size_for_anomaly, shuffle=False)
        with torch.no_grad(): 
            for x_batch_tuple in anomaly_loader: 
                x_batch = x_batch_tuple[0].to(DEVICE)
                outputs_scaled_batch = self.model(x_batch) 
                all_predictions_scaled_list.append(outputs_scaled_batch.cpu().numpy())
        if not all_predictions_scaled_list:
            print("错误: 模型未能对异常检测数据生成任何预测输出。异常检测中止。")
            return {}
        predictions_scaled_full_horizon = np.concatenate(all_predictions_scaled_list, axis=0)
        predictions_scaled_one_step = predictions_scaled_full_horizon[:, 0, :] 
        predicted_original_one_step = np.zeros_like(predictions_scaled_one_step)
        for i, col_name in enumerate(model_target_cols):
            scaler = self.target_scalers[col_name] 
            pred_col_scaled_reshaped = predictions_scaled_one_step[:, i].reshape(-1, 1) 
            predicted_original_one_step[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten()
        actual_original_one_step = y_actual_anomaly_np[:, 0, :]
        errors = np.abs(actual_original_one_step - predicted_original_one_step) 
        anomaly_reports = {}
        print("\n--- 异常数据检测详细报告 ---")
        num_sequences_generated = X_anomaly_np.shape[0]
        if len(df_for_sequences.index) >= self.model_config['look_back'] + num_sequences_generated:
            base_timestamps_for_y = df_for_sequences.index[self.model_config['look_back'] : self.model_config['look_back'] + num_sequences_generated]
        else: 
            print("警告: 用于异常检测的时间戳序列长度不足，可能导致绘图或报告的时间不准确。将使用简单范围索引。")
            base_timestamps_for_y = pd.RangeIndex(start=0, stop=num_sequences_generated, step=1)
        for i, col_name in enumerate(model_target_cols):
            col_errors = errors[:, i] 
            if len(col_errors) == 0: 
                print(f"\n目标污染物: {col_name} - 无有效误差数据可供分析。")
                anomaly_reports[col_name] = {'count': 0, 'threshold': np.nan, 'timestamps': [], 'details': []}
                continue
            mean_error = np.mean(col_errors)
            std_error = np.std(col_errors)
            threshold = mean_error + current_threshold_factor * std_error if std_error > 1e-9 else mean_error + 1e-9 
            anomaly_flags_col = col_errors > threshold
            current_col_anomaly_indices_relative = np.where(anomaly_flags_col)[0] 
            anomaly_timestamps_col = [base_timestamps_for_y[k] for k in current_col_anomaly_indices_relative if k < len(base_timestamps_for_y)]
            anomaly_details_list = []
            for k_idx, rel_idx in enumerate(current_col_anomaly_indices_relative):
                if rel_idx < len(actual_original_one_step): 
                    anomaly_details_list.append({
                        'timestamp': base_timestamps_for_y[rel_idx] if rel_idx < len(base_timestamps_for_y) else "索引越界",
                        'actual_value': actual_original_one_step[rel_idx, i],
                        'predicted_value': predicted_original_one_step[rel_idx, i],
                        'error': col_errors[rel_idx]
                    })
            anomaly_reports[col_name] = {
                'count': len(anomaly_timestamps_col),
                'threshold_value': threshold, 
                'mean_error': mean_error, 
                'std_dev_error': std_error,
                'relative_indices_in_sequence': current_col_anomaly_indices_relative.tolist(),
                'timestamps_of_anomalies': anomaly_timestamps_col,
                'anomaly_details': anomaly_details_list
            }
            print(f"\n目标污染物: {col_name}")
            print(f"  统计信息: 平均绝对误差={mean_error:.4f}, 误差标准差={std_error:.4f}")
            print(f"  异常判定阈值 (误差 > {threshold:.4f})")
            if len(anomaly_timestamps_col) > 0:
                print(f"  检测到 {len(anomaly_timestamps_col)} 个潜在异常点:")
                for detail in anomaly_details_list[:5]: 
                    print(f"    - 时间: {detail['timestamp']}, 实际值: {detail['actual_value']:.2f}, "
                          f"预测值: {detail['predicted_value']:.2f}, 误差: {detail['error']:.2f}")
                if len(anomaly_timestamps_col) > 5: 
                    print(f"    ...等 (共 {len(anomaly_timestamps_col)} 个异常点)")
            else: 
                print("  在此阈值下，未检测到异常点。")
            plot_save_prefix_target = os.path.join(self.artifacts_dir, f"anomaly_detection_report_{col_name}") 
            plot_anomalies(
                timestamps_or_indices=base_timestamps_for_y, 
                actual_values=actual_original_one_step[:, i], 
                anomaly_indices=current_col_anomaly_indices_relative, 
                target_name=col_name,
                save_path_prefix=plot_save_prefix_target, 
                title_suffix=f"异常点检测 (阈值因子={current_threshold_factor})"
            )
            print(f"  '{col_name}' 的异常点可视化图表已保存至: {plot_save_prefix_target}_anomalies.png")
        print("\n--- 异常数据检测流程结束 ---")
        return anomaly_reports

if __name__ == "__main__":
    print(f"系统当前使用的计算设备: {DEVICE}")
    action = input("您好！请选择要执行的操作： (1: 训练新模型, 2: 使用现有模型进行预测, 3: 使用现有模型检测异常数据): ").strip()
    if action == '1':
        print("\n--- 您已选择：1. 训练新模型 ---")
        custom_artifacts_dir = input(f"请输入模型文件及相关组件的保存目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        train_data_path = input(f"请输入训练数据文件的完整路径 (例如: data/my_training_data.xlsx, 默认为 '{DEFAULT_FILE_PATH}'): ").strip() or DEFAULT_FILE_PATH
        if not os.path.exists(train_data_path):
            print(f"错误: 训练数据文件 '{train_data_path}' 未找到。请检查路径是否正确。")
        else:
            system = AQISystem(artifacts_dir=custom_artifacts_dir) 
            print(f"开始使用数据 '{train_data_path}' 在目录 '{custom_artifacts_dir}' 中训练新模型...")
            system.train_new_model(train_data_path=train_data_path)
            print("新模型训练流程已完成。")
    elif action == '2':
        print("\n--- 您已选择：2. 使用现有模型进行预测 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        input_data_source = input("请输入用于预测的输入数据文件路径 (CSV 或 Excel格式，应包含足够的历史数据): ").strip()
        if not os.path.exists(input_data_source):
            print(f"错误: 预测输入数据文件 '{input_data_source}' 未找到。请检查路径。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir) 
                print(f"开始使用位于 '{custom_artifacts_dir}' 的模型对数据 '{input_data_source}' 进行预测...")
                system.predict_with_existing_model(input_data_path_or_df=input_data_source)
                print("预测流程已完成。")
            except FileNotFoundError as e: 
                print(f"文件未找到错误 (预测初始化或执行时): {e}。请确认模型工件目录和输入数据文件路径是否正确。")
            except ValueError as e: 
                print(f"数据或配置值错误 (预测流程): {e}")
            except Exception as e: 
                print(f"预测过程中发生未知类型的错误: {e}")
                import traceback; traceback.print_exc()
    elif action == '3':
        print("\n--- 您已选择：3. 使用现有模型检测异常数据 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        anomaly_data_path = input("请输入用于异常检测的数据文件路径 (CSV 或 Excel格式，应包含实际观测值): ").strip()
        threshold_factor_str = input(f"请输入异常检测的阈值敏感度因子 (例如输入 3.0 表示均值+3倍标准差，默认为 {DEFAULT_ANOMALY_THRESHOLD_FACTOR}): ").strip()
        custom_threshold_factor = None 
        if threshold_factor_str: 
            try: 
                custom_threshold_factor = float(threshold_factor_str)
                if custom_threshold_factor <= 0:
                     print("警告: 阈值因子必须为正数。将使用默认值。")
                     custom_threshold_factor = None 
            except ValueError: 
                print("警告: 输入的阈值因子不是有效的数字。将使用默认值。")
        if not os.path.exists(anomaly_data_path):
            print(f"错误: 异常检测数据文件 '{anomaly_data_path}' 未找到。请检查路径。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir) 
                print(f"开始使用位于 '{custom_artifacts_dir}' 的模型对数据 '{anomaly_data_path}' 进行异常检测...")
                anomaly_report = system.detect_anomalies(data_path_or_df=anomaly_data_path, threshold_factor=custom_threshold_factor)
                print("异常数据检测流程已完成。详细报告和图表已生成（如果检测到异常）。")
            except FileNotFoundError as e: 
                print(f"文件未找到错误 (异常检测初始化或执行时): {e}。请确认模型工件目录和数据文件路径。")
            except ValueError as e: 
                print(f"数据或配置值错误 (异常检测流程): {e}")
            except Exception as e: 
                print(f"异常检测过程中发生未知类型的错误: {e}")
                import traceback; traceback.print_exc()
    else:
        print("无效的操作选择。请输入 '1', '2', 或 '3'。程序即将退出。")
    print("\nAQI预测与异常检测系统运行结束。")
