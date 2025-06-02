# ==============================================================================
# 主要功能：
# 本脚本实现了一个基于Transformer模型的多目标空气质量指标（AQI及多种污染物浓度）预测与异常检测系统。
# 功能包括：
# 1. 数据加载与高级预处理：
#    - 时间戳处理与索引设置。
#    - 稳健的数值转换和缺失值（NaN）填充策略。
#    - 基于IQR的异常值检测与插值修复（训练时）。
#    - 特征工程：AQI计算、周期性特征、滞后特征。
# 2. 模型训练 (通过 AQISystem -> ModelTrainer)：
#    - 使用PyTorch实现的Transformer模型架构。
#    - 支持多目标同时预测。
#    - 采用Optuna进行超参数优化。
#    - 学习率调度器和早停机制。
# 3. 模型预测 (通过 AQISystem -> ModelPredictor logic)：
#    - 加载已训练的模型和相关组件。
#    - 对新的输入数据进行预测。
# 4. 异常检测 (通过 AQISystem)：
#    - 使用已训练的模型识别输入数据中的异常点。
#    - 基于预测误差和统计阈值进行判断。
#    - 可视化异常点。
# 5. 用户交互：
#    - 脚本运行时会询问用户进行模型训练、预测还是异常检测。
#
# 设计思路与优化：
# - 统一入口：AQISystem 类作为所有操作的统一入口。
# - 模块化：ModelTrainer 和 ModelPredictor (其逻辑被AQISystem部分集成或调用) 负责特定任务。
# - 鲁棒性：增强了数据预处理的鲁棒性。
# - 准确度与实用性：结合了预测与异常检测能力。
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
import joblib # 用于保存和加载scaler
import json # 用于保存模型配置

import optuna # 用于超参数优化

# --- Matplotlib 中文显示设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"设置中文字体失败: {e}。绘图标签可能无法正确显示中文。")

# --- 全局参数设置 ---
DEFAULT_FILE_PATH = '南京_AQI_Data.xlsx' 
DEFAULT_LOOK_BACK = 24 
DEFAULT_HORIZON = 72   
DEFAULT_TARGET_COL_NAMES = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
DEFAULT_ALL_AVAILABLE_COL_NAMES = [ 
    'date', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
    'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 
    'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'
]
DEFAULT_PRIMARY_TARGET_COL_NAME = 'AQI' 
DEFAULT_BATCH_SIZE = 32 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

MODEL_ARTIFACTS_DIR = "model_artifacts_advanced" 
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model_adv.pth" 
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler_adv.pkl" 
TARGET_SCALERS_SAVE_NAME = "aqi_target_scalers_adv.pkl" 
MODEL_CONFIG_SAVE_NAME = "model_config_adv.json" 

DEFAULT_FULL_TRAIN_EPOCHS = 200  
DEFAULT_N_OPTUNA_TRIALS = 150    
DEFAULT_OPTUNA_EPOCHS = 30     
DEFAULT_EARLY_STOPPING_PATIENCE = 20 
DEFAULT_MIN_DELTA = 0.00001 
DEFAULT_ANOMALY_THRESHOLD_FACTOR = 3.0 # 异常检测阈值因子 (N倍标准差)

# --- AQI 计算相关的常量 ---
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


# --- 工具函数 ---
def set_seed(seed_value=42):
    """设置全局随机种子以保证实验结果的可复现性。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def calculate_iaqi(Cp, pollutant_key):
    """计算单个污染物的个体空气质量指数 (IAQI)。"""
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
    """根据输入的污染物浓度DataFrame计算AQI，并确定首要污染物。"""
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
    """从时间序列数据创建输入序列 (X) 和输出序列 (y)。"""
    X_list, y_list = [], []
    missing_feature_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_feature_cols: raise ValueError(f"数据缺少特征列: {missing_feature_cols}.")
    if not is_predict:
        missing_target_cols = [col for col in target_col_names if col not in data_df.columns]
        if missing_target_cols: raise ValueError(f"数据缺少目标列: {missing_target_cols}.")

    data_features_np = data_df[feature_cols].values
    if not is_predict: data_targets_np = data_df[target_col_names].values
    
    num_samples = len(data_features_np)
    if is_predict: 
        num_possible_sequences = num_samples - look_back + 1
    else: 
        num_possible_sequences = num_samples - look_back - horizon + 1

    if num_possible_sequences <= 0: 
        return np.array(X_list), np.array(y_list) 

    for i in range(num_possible_sequences):
        X_list.append(data_features_np[i : i + look_back])
        if not is_predict:
            y_list.append(data_targets_np[i + look_back : i + look_back + horizon, :])
            
    X_arr = np.array(X_list) if X_list else np.array([])
    if is_predict: return X_arr
    y_arr = np.array(y_list) if y_list else np.array([])
    
    if y_arr.size == 0 and X_arr.size > 0 and not is_predict : 
        print("警告: 创建了X序列，但对应的y序列为空（数据长度不足）。")
        return np.array([]), np.array([])
        
    return X_arr, y_arr


def plot_training_loss(train_losses, val_losses, save_path, title_prefix=""):
    """绘制并保存训练和验证损失曲线。"""
    plt.figure(figsize=(10, 6)); plt.plot(train_losses, label='训练损失'); plt.plot(val_losses, label='验证损失')
    plt.title(f'{title_prefix}损失变化'); plt.xlabel('Epoch'); plt.ylabel('损失(MSE)'); plt.legend(); plt.grid(True)
    plt.savefig(save_path); plt.close()

def plot_predictions_vs_actual(actual, predicted, target_name, save_path_prefix, title_suffix="实际 vs. 预测"):
    """为单个目标绘制并保存实际值与预测值的对比图。"""
    plt.figure(figsize=(15, 7)); actual_flat = actual.flatten(); predicted_flat = predicted.flatten()
    plt.plot(actual_flat, label=f'实际{target_name}', alpha=0.7); plt.plot(predicted_flat, label=f'预测{target_name}', linestyle='--', alpha=0.7)
    plt.title(f'{target_name} - {title_suffix}'); plt.xlabel('时间步'); plt.ylabel(target_name); plt.legend(); plt.grid(True)
    plt.savefig(f"{save_path_prefix}_{target_name}_predictions.png"); plt.close()

def plot_anomalies(timestamps_or_indices, actual_values, anomaly_indices, target_name, save_path_prefix, title_suffix="异常点"):
    """为单个目标绘制并保存带有异常点标记的时间序列图。"""
    plt.figure(figsize=(15, 7))
    actual_flat = actual_values.flatten() 

    plt.plot(timestamps_or_indices, actual_flat, label=f'实际{target_name}', alpha=0.7)
    
    valid_anomaly_indices = np.array([idx for idx in anomaly_indices if 0 <= idx < len(actual_flat)], dtype=int)

    if len(valid_anomaly_indices) > 0:
        if isinstance(timestamps_or_indices, (pd.DatetimeIndex, pd.RangeIndex)):
            plot_x_values = np.array(timestamps_or_indices)
        else: 
            plot_x_values = np.asarray(timestamps_or_indices) # Ensure it's an array

        anomaly_x = plot_x_values[valid_anomaly_indices]
        anomaly_y = actual_flat[valid_anomaly_indices]
        plt.scatter(anomaly_x, anomaly_y, color='red', label='异常点', marker='o', s=50, zorder=5)

    plt.title(f'{target_name} - {title_suffix}')
    plt.xlabel('时间/索引') 
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    save_full_path = f"{save_path_prefix}_anomalies.png" 
    plt.savefig(save_full_path)
    plt.close()


def detect_anomalies_iqr_and_impute(df, column_names, factor=1.5, interpolation_method='time'):
    """使用IQR方法检测指定列中的异常点，并进行插值填充。"""
    df_cleaned = df.copy(); print("开始数据异常值检测和插值填充...")
    for col_name in column_names:
        if col_name in df_cleaned.columns:
            if not pd.api.types.is_numeric_dtype(df_cleaned[col_name]):
                print(f"警告: 列 '{col_name}' 不是数值类型，跳过异常处理。")
                continue
            original_nan_count = df_cleaned[col_name].isna().sum()
            Q1 = df_cleaned[col_name].quantile(0.25); Q3 = df_cleaned[col_name].quantile(0.75)
            IQR = Q3 - Q1
            if pd.notna(IQR) and IQR > 1e-6: 
                lower_bound = Q1 - factor * IQR; upper_bound = Q3 + factor * IQR
                outlier_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
                num_outliers = outlier_mask.sum()
                if num_outliers > 0: 
                    print(f"列'{col_name}'检测到{num_outliers}个异常值。尝试插值..."); 
                    df_cleaned.loc[outlier_mask, col_name] = np.nan 
            else:
                print(f"列'{col_name}'的IQR为0或无效，跳过基于IQR的异常值标记。")
            if isinstance(df_cleaned.index, pd.DatetimeIndex): 
                try: df_cleaned[col_name] = df_cleaned[col_name].interpolate(method=interpolation_method, limit_direction='both')
                except Exception as e: print(f"列'{col_name}'时间插值失败:{e}。尝试线性插值。"); df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            else: 
                df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            if df_cleaned[col_name].isna().sum() > 0 : 
                median_val = df_cleaned[col_name].median() 
                fill_value = median_val if pd.notna(median_val) else 0 
                df_cleaned[col_name] = df_cleaned[col_name].fillna(fill_value)
                print(f"列'{col_name}'的剩余NaN已用中位数/0 ({fill_value:.2f})填充。")
        else: print(f"警告:列'{col_name}'未找到，跳过异常处理。")
    print("异常值处理完成。"); return df_cleaned

class TimeSeriesDataset(TensorDataset):
    """自定义PyTorch数据集，用于时间序列数据。"""
    def __init__(self, X, y): 
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor)

class AQITransformer(nn.Module):
    """基于Transformer Encoder的多目标时间序列预测模型。"""
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, num_target_features, norm_first=True):
        super(AQITransformer, self).__init__(); self.d_model=d_model; self.horizon=horizon; self.num_target_features=num_target_features
        self.input_embedding = nn.Linear(num_features, d_model) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers) 
        self.output_layer = nn.Linear(d_model, horizon * num_target_features) 
    
    def forward(self, src):
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model) 
        seq_len = src_embedded.size(1)
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 != 0: pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)] 
        else: pe[:, 1::2] = torch.cos(position * div_term)
        src_pos_encoded = src_embedded + pe.unsqueeze(0) 
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded) 
        encoder_output = self.transformer_encoder(src_pos_encoded) 
        prediction_input = encoder_output[:, -1, :] 
        output_flat = self.output_layer(prediction_input) 
        output = output_flat.view(output_flat.size(0), self.horizon, self.num_target_features)
        return output

class ModelTrainer:
    """该类封装了模型训练的整个流程。"""
    def __init__(self, config):
        self.config = config # config 由 AQISystem 传入
        # model_artifacts_dir 应该由 config 控制，AQISystem 会确保它存在
        # set_seed() 应该在 AQISystem 初始化时调用，或在此处按需调用
        self.all_feature_columns_for_sequence = [] 
        self.feature_scaler = None 
        self.target_scalers = {}   

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True):
        """核心的数据加载和预处理函数。"""
        print(f"开始加载和预处理数据从: {file_path}...")
        try: df = pd.read_csv(file_path)
        except Exception:
            try: df = pd.read_excel(file_path)
            except Exception as e_excel: print(f"读取Excel或CSV均失败: {e_excel}"); raise

        if 'date' in df.columns and 'hour' in df.columns: 
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(int).astype(str).str.zfill(2), format='%Y%m%d%H')
            df = df.set_index('timestamp').drop(columns=['date', 'hour'], errors='ignore')
        elif 'Time' in df.columns: 
             df['timestamp'] = pd.to_datetime(df['Time']); df = df.set_index('timestamp').drop(columns=['Time'], errors='ignore')
        else: 
            try: df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]); df = df.set_index(df.columns[0])
            except Exception: raise ValueError("无法自动设置时间索引。")

        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        feature_candidate_cols = [col for col in df.columns if col not in self.config['target_col_names']]
        for col in feature_candidate_cols: 
            if df[col].isnull().any():
                df[col] = df[col].ffill().bfill().fillna(0) 
        df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
        if df.empty: raise ValueError("数据在初步NaN处理后DataFrame为空。")

        if fit_scalers: 
            df = detect_anomalies_iqr_and_impute(df, self.config['target_col_names'])
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names']) 
        if df.empty: raise ValueError("数据在异常值处理后DataFrame为空。")

        df = calculate_aqi_from_pollutants(df.copy()) 
        if 'AQI_calculated' in df.columns and df['AQI_calculated'].isnull().any():
            df['AQI_calculated'] = df['AQI_calculated'].fillna(0)

        new_cyclical_features = pd.DataFrame(index=df.index)
        if isinstance(df.index, pd.DatetimeIndex):
            idx = df.index
            new_cyclical_features['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0)
            new_cyclical_features['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['month_sin'] = np.sin(2*np.pi*idx.month/12.0) 
            new_cyclical_features['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
        df = pd.concat([df, new_cyclical_features], axis=1)
        
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
            df = df.dropna(subset=lag_cols_created_names, how='any') 
        
        if df.empty: raise ValueError("数据在创建滞后特征后DataFrame为空。")

        self.all_feature_columns_for_sequence = [col for col in df.columns if col not in self.config['target_col_names']]
        if 'AQI_calculated' in df.columns and 'AQI_calculated' not in self.config['target_col_names']:
            if 'AQI_calculated' not in self.all_feature_columns_for_sequence:
                 self.all_feature_columns_for_sequence.append('AQI_calculated')
        elif 'AQI_calculated' in self.all_feature_columns_for_sequence and 'AQI_calculated' in self.config['target_col_names']:
             print("警告: 'AQI_calculated' 同时被定义为目标和特征。")
        
        self.all_feature_columns_for_sequence = sorted(list(set(self.all_feature_columns_for_sequence))) 

        missing_targets_for_scaling = [tc for tc in self.config['target_col_names'] if tc not in df.columns]
        if missing_targets_for_scaling: raise ValueError(f"目标列在缩放前未找到: {missing_targets_for_scaling}.")
        
        if fit_scalers:
            self.feature_scaler = StandardScaler() 
            current_features_to_scale = [f_col for f_col in self.all_feature_columns_for_sequence if f_col in df.columns] 
            
            for f_col in current_features_to_scale: 
                if not pd.api.types.is_numeric_dtype(df[f_col]):
                    df[f_col] = pd.to_numeric(df[f_col], errors='coerce')
                    if df[f_col].isnull().any(): 
                        df[f_col] = df[f_col].fillna(0)
                    if not pd.api.types.is_numeric_dtype(df[f_col]): 
                         offending_values = [item for item in df[f_col].unique() if not isinstance(item, (int, float, np.number))]
                         raise ValueError(f"特征列 '{f_col}' 无法转换为数值类型进行缩放。问题值示例: {offending_values[:5]}")
            
            if not current_features_to_scale: 
                raise ValueError("没有有效的特征列可用于缩放。")

            df[current_features_to_scale] = self.feature_scaler.fit_transform(df[current_features_to_scale])
            joblib.dump(self.feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))
            
            self.target_scalers = {} 
            for col_name in self.config['target_col_names']:
                if not pd.api.types.is_numeric_dtype(df[col_name]): 
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0) 
                scaler = StandardScaler(); df[[col_name]] = scaler.fit_transform(df[[col_name]])
                self.target_scalers[col_name] = scaler
            joblib.dump(self.target_scalers, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALERS_SAVE_NAME))
            print("特征和目标缩放器已拟合和保存。")
        return df

    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, trial=None):
        """核心模型训练循环。"""
        best_val_loss = float('inf'); epochs_no_improve = 0; best_model_state = None
        train_losses_epoch, val_losses_epoch = [], []
        primary_target_idx = self.config['target_col_names'].index(self.config['primary_target_col_name'])
        for epoch in range(epochs):
            model.train(); running_train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE); optimizer.zero_grad()
                outputs = model(X_batch); loss = criterion(outputs, y_batch); loss.backward(); optimizer.step()
                running_train_loss += loss.item() * X_batch.size(0)
            epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
            train_losses_epoch.append(epoch_train_loss)
            model.eval(); running_val_loss = 0.0; running_primary_target_val_loss = 0.0
            if len(val_loader.dataset) > 0:
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE); outputs = model(X_batch)
                        loss = criterion(outputs, y_batch); running_val_loss += loss.item() * X_batch.size(0)
                        primary_target_loss = criterion(outputs[:, :, primary_target_idx], y_batch[:, :, primary_target_idx])
                        running_primary_target_val_loss += primary_target_loss.item() * X_batch.size(0)
                epoch_val_loss = running_val_loss / len(val_loader.dataset); epoch_primary_target_val_loss = running_primary_target_val_loss / len(val_loader.dataset)
            else: epoch_val_loss = float('inf'); epoch_primary_target_val_loss = float('inf')
            val_losses_epoch.append(epoch_val_loss)
            print(f"Epoch [{epoch+1}/{epochs}], LR: {optimizer.param_groups[0]['lr']:.7f}, Train Loss: {epoch_train_loss:.6f}, Val Loss (Overall): {epoch_val_loss:.6f}, Val Loss ({self.config['primary_target_col_name']}): {epoch_primary_target_val_loss:.6f}")
            if scheduler: scheduler.step(epoch_primary_target_val_loss) 
            if trial:
                trial.report(epoch_primary_target_val_loss, epoch) 
                if trial.should_prune(): print("Optuna trial pruned."); raise optuna.exceptions.TrialPruned()
            if epoch_primary_target_val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = epoch_primary_target_val_loss; epochs_no_improve = 0; best_model_state = copy.deepcopy(model.state_dict())
                if trial is None: torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME)); print(f"Val loss ({self.config['primary_target_col_name']}) improved. Model saved at epoch {epoch+1}.")
            else: epochs_no_improve += 1
            if epochs_no_improve >= self.config['early_stopping_patience'] and len(val_loader.dataset) > 0 : 
                print(f"Early stopping triggered at epoch {epoch+1} based on {self.config['primary_target_col_name']}.");
                if best_model_state: model.load_state_dict(best_model_state); break
        if best_model_state and trial is None: 
             model.load_state_dict(best_model_state) 
             torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
             print(f"Final best model state from training saved to {MODEL_STATE_SAVE_NAME}")
        return model, train_losses_epoch, val_losses_epoch, best_val_loss

    def _objective_optuna(self, trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features):
        """Optuna的目标函数。"""
        lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True) 
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512]) 
        possible_num_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and d_model >= h]
        if not possible_num_heads: raise optuna.exceptions.TrialPruned("No valid number of heads for d_model.")
        num_heads = trial.suggest_categorical('num_heads', possible_num_heads)
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8) 
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 6) 
        dim_feedforward = d_model * dim_feedforward_factor
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4) 
        norm_first = trial.suggest_categorical('norm_first', [True, False])
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.7)
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)
        model = AQITransformer(num_features=num_input_features, d_model=d_model, nhead=num_heads, num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, norm_first=norm_first, horizon=self.config['horizon'], num_target_features=len(self.config['target_col_names'])).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience) 
        criterion = nn.MSELoss() 
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 for simplicity in some environments
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        print(f"\nOptuna Trial {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.3f}, norm_first={norm_first}, wd={weight_decay:.7f}, sch_factor={scheduler_factor:.2f}, sch_patience={scheduler_patience}")
        _, _, _, best_val_loss_trial = self._train_model_core(model, train_loader, val_loader, criterion, optimizer, scheduler, self.config['optuna_epochs'], trial)
        return best_val_loss_trial

    def run_training_pipeline(self):
        """执行完整的模型训练流程。"""
        print("--- 开始模型训练流程 ---")
        df_processed = self._load_and_preprocess_data_core(self.config['file_path'], fit_scalers=True)
        num_input_features_for_model = len(self.all_feature_columns_for_sequence) 
        X_initial, y_initial = create_sequences(df_processed, self.config['look_back'], self.config['horizon'], self.config['target_col_names'], self.all_feature_columns_for_sequence)
        if X_initial.size == 0: print("创建序列后数据为空，终止训练。"); return
        total_samples = X_initial.shape[0]; train_idx_end = int(total_samples * 0.7); val_idx_end = int(total_samples * 0.85)
        X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]; X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
        X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:]
        print(f"数据集大小: 训练={X_train_np.shape[0]}, 验证={X_val_np.shape[0]}, 测试={X_test_np.shape[0]}")
        if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0: print("训练集或验证集为空。"); return
        print("\n开始Optuna超参数优化...")
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=self.config['optuna_epochs'], reduction_factor=3), sampler=optuna.samplers.TPESampler(seed=42)) 
        study.optimize(lambda trial: self._objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features_for_model), n_trials=self.config['n_optuna_trials'], timeout=3600*6) 
        best_hyperparams = study.best_params
        print(f"最佳试验验证损失 ({self.config['primary_target_col_name']}): {study.best_value:.6f}\n最佳超参数: {best_hyperparams}")
        print("\n使用最佳超参数训练最终模型...")
        final_model_arch_params = {'d_model': best_hyperparams['d_model'], 'nhead': best_hyperparams['num_heads'], 'num_encoder_layers': best_hyperparams['num_encoder_layers'],
            'dim_feedforward': best_hyperparams['d_model'] * best_hyperparams['dim_feedforward_factor'], 'dropout': best_hyperparams['dropout_rate'], 'norm_first': best_hyperparams['norm_first']}
        final_model = AQITransformer(num_features=num_input_features_for_model, **final_model_arch_params, horizon=self.config['horizon'], num_target_features=len(self.config['target_col_names'])).to(DEVICE)
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_hyperparams['learning_rate'], weight_decay=best_hyperparams.get('weight_decay', 0.0))
        final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=best_hyperparams.get('scheduler_factor', 0.5), patience=best_hyperparams.get('scheduler_patience', 7)) 
        criterion = nn.MSELoss()
        final_model, train_losses, val_losses, _ = self._train_model_core(final_model, final_train_loader, final_val_loader, criterion, final_optimizer, final_scheduler, self.config['full_train_epochs'])
        plot_training_loss(train_losses, val_losses, os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"), title_prefix="最终模型 (Overall)")
        print(f"最终模型训练完成并保存。")
        model_config_to_save = {'model_architecture': final_model_arch_params, 'look_back': self.config['look_back'], 'horizon': self.config['horizon'],
            'target_col_names': self.config['target_col_names'], 'primary_target_col_name': self.config['primary_target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence, # 保存这个重要列表
            'num_input_features_for_model': num_input_features_for_model,
            'num_target_features': len(self.config['target_col_names']), 'optuna_best_params': best_hyperparams } 
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w', encoding='utf-8') as f: json.dump(model_config_to_save, f, indent=4, ensure_ascii=False)
        print(f"模型配置已保存。")
        if X_test_np.shape[0] > 0: self.evaluate_trained_model(final_model, X_test_np, y_test_np, criterion)

    def evaluate_trained_model(self, model, X_test_np, y_test_np, criterion):
        """评估训练好的模型在测试集上的性能。"""
        print("\n评估最终模型在测试集上的性能...")
        test_loader = DataLoader(TimeSeriesDataset(X_test_np, y_test_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        model.eval(); all_preds_scaled, all_targets_scaled = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(DEVICE)); all_preds_scaled.append(outputs.cpu().numpy()); all_targets_scaled.append(y_batch.numpy())
        if not all_preds_scaled: print("测试集上没有生成预测。"); return
        preds_scaled_np = np.concatenate(all_preds_scaled, axis=0); targets_scaled_np = np.concatenate(all_targets_scaled, axis=0)
        actual_orig_all_targets = np.zeros_like(targets_scaled_np); predicted_orig_all_targets = np.zeros_like(preds_scaled_np)
        for i, col_name in enumerate(self.config['target_col_names']):
            scaler = self.target_scalers[col_name]
            actual_orig_all_targets[:, :, i] = scaler.inverse_transform(targets_scaled_np[:, :, i])
            predicted_orig_all_targets[:, :, i] = scaler.inverse_transform(preds_scaled_np[:, :, i])
        print("\n各目标在测试集上的评估指标:")
        for i, col_name in enumerate(self.config['target_col_names']):
            actual_col = actual_orig_all_targets[:, :, i].flatten(); predicted_col = predicted_orig_all_targets[:, :, i].flatten()
            mae = mean_absolute_error(actual_col, predicted_col); rmse = np.sqrt(mean_squared_error(actual_col, predicted_col)); r2 = r2_score(actual_col, predicted_col)
            print(f"  {col_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            plot_save_prefix = os.path.join(self.config['model_artifacts_dir'], f"final_model_test") 
            plot_predictions_vs_actual(actual_orig_all_targets[:, :, i], predicted_orig_all_targets[:, :, i], col_name, plot_save_prefix, title_suffix="测试集")
        print("\n注意: 中国AQI是根据PM2.5, PM10, SO2, NO2, O3, CO等污染物的浓度计算得出的综合指数。")

class ModelPredictor: # 主要逻辑会被 AQISystem 集成或调用
    """封装加载已训练模型并进行预测的逻辑。"""
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir; self.model = None; self.feature_scaler = None; self.target_scalers = {}; self.model_config = None
        self._load_artifacts() # 在初始化时加载

    def _load_artifacts(self):
        print(f"从 {self.artifacts_dir} 加载模型及相关组件 (ModelPredictor)...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME); model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME); ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)
        if not all(os.path.exists(p) for p in [config_path, model_path, fs_path, ts_path]):
            missing_files = [p for p in [config_path, model_path, fs_path, ts_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"一个或多个必要的模型文件在 '{self.artifacts_dir}' 中未找到: {missing_files}。")
        with open(config_path, 'r', encoding='utf-8') as f: self.model_config = json.load(f)
        self.feature_scaler = joblib.load(fs_path); self.target_scalers = joblib.load(ts_path)
        self.model = AQITransformer(num_features=self.model_config['num_input_features_for_model'], **self.model_config['model_architecture'], 
                                   horizon=self.model_config['horizon'], num_target_features=self.model_config['num_target_features']).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)); self.model.eval(); print("模型及相关组件加载成功 (ModelPredictor)。")

    def _preprocess_input_for_prediction(self, df_raw):
        """为预测预处理输入数据，与训练过程保持一致。"""
        print("预处理输入数据进行预测 (ModelPredictor)...")
        df_processed = df_raw.copy()
        if not isinstance(df_processed.index, pd.DatetimeIndex): 
            if 'Time' in df_processed.columns: df_processed['timestamp'] = pd.to_datetime(df_processed['Time']); df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                 df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                 df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                 df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else:
                try:
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                        df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0]); df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: raise ValueError(f"无法为预测数据设置时间索引: {e}。")
        
        for col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 
        
        # 特征列的NaN填充
        feature_cols_in_df = [col for col in df_processed.columns if col not in self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)]
        for col in feature_cols_in_df: 
            if df_processed[col].isnull().any(): df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        # AQI_calculated 特征 (如果模型需要)
        if 'AQI_calculated' in self.model_config['all_feature_columns_for_sequence']:
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi: 
                df_processed = calculate_aqi_from_pollutants(df_processed.copy())
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                    df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            else: 
                print("警告: 预测数据缺少计算'AQI_calculated'的列，用0填充。"); df_processed['AQI_calculated'] = 0 
        
        # 周期性特征
        new_cyclical_features_pred = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex): 
            idx = df_processed.index
            new_cyclical_features_pred['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features_pred['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features_pred['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features_pred['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features_pred['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features_pred['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
        df_processed = pd.concat([df_processed, new_cyclical_features_pred], axis=1)
        
        # 滞后特征
        lag_features_to_recreate = [f for f in self.model_config['all_feature_columns_for_sequence'] if "_lag_" in f] 
        lag_series_list_pred = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_"); 
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]; lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns: 
                    lag_series_list_pred.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: 
                    lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: 
                lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        
        if lag_series_list_pred:
             df_processed = pd.concat([df_processed] + lag_series_list_pred, axis=1)

        # 合并滞后特征后，再次填充所有列的NaN
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        # 确保所有模型期望的特征都存在
        expected_features_from_config = self.model_config['all_feature_columns_for_sequence'] 
        df_for_scaling = pd.DataFrame(index=df_processed.index) # 创建一个空的DataFrame来收集特征
        for f_col in expected_features_from_config:
            if f_col not in df_processed.columns: 
                print(f"警告: 预测数据中缺少特征 '{f_col}'。用0填充。"); 
                df_for_scaling[f_col] = 0 # 添加全0列
            else:
                df_for_scaling[f_col] = df_processed[f_col] # 从处理后的数据中获取
        
        # 特征缩放前确保数值类型
        for f_col in expected_features_from_config: 
            df_for_scaling[f_col] = pd.to_numeric(df_for_scaling[f_col], errors='coerce').fillna(0)
            if not pd.api.types.is_numeric_dtype(df_for_scaling[f_col]):
                try: df_for_scaling[f_col] = df_for_scaling[f_col].astype(float)
                except ValueError as e_astype:
                    offending_values = [item for item in df_for_scaling[f_col].unique() if not isinstance(item, (int, float, np.number))]
                    raise ValueError(f"无法将预测特征列 '{f_col}' 转为浮点数。错误值示例: {offending_values[:5]}. 原错误: {e_astype}")

        df_for_scaling_transformed = self.feature_scaler.transform(df_for_scaling[expected_features_from_config]) # 只选择期望的特征列进行变换
        df_scaled_features = pd.DataFrame(df_for_scaling_transformed, columns=expected_features_from_config, index=df_for_scaling.index)
        return df_scaled_features

    def predict(self, input_data_path_or_df):
        if isinstance(input_data_path_or_df, str):
            try: df_raw = pd.read_csv(input_data_path_or_df)
            except Exception: df_raw = pd.read_excel(input_data_path_or_df)
        elif isinstance(input_data_path_or_df, pd.DataFrame): df_raw = input_data_path_or_df.copy()
        else: raise ValueError("输入数据必须是文件路径(CSV/Excel)或Pandas DataFrame。")
        
        last_known_timestamp = None # 尝试确定最后一个时间戳
        # (与原脚本中类似的逻辑来确定 last_known_timestamp)
        if isinstance(df_raw.index, pd.DatetimeIndex) and not df_raw.empty: last_known_timestamp = df_raw.index[-1]
        elif not df_raw.empty:
            time_cols = ['Time', 'timestamp', 'Datetime']; 
            for tc in time_cols:
                if tc in df_raw.columns: last_known_timestamp = pd.to_datetime(df_raw[tc].iloc[-1]); break
            if last_known_timestamp is None and 'date' in df_raw.columns and 'hour' in df_raw.columns:
                date_val = str(df_raw['date'].iloc[-1])
                hour_val = str(int(df_raw['hour'].iloc[-1])).zfill(2) 
                last_dt_str = date_val + hour_val
                last_known_timestamp = pd.to_datetime(last_dt_str, format='%Y%m%d%H')
            elif last_known_timestamp is None: 
                 try:
                     if not pd.api.types.is_numeric_dtype(df_raw.iloc[:,0]): last_known_timestamp = pd.to_datetime(df_raw.iloc[-1, 0])
                 except: pass
        if last_known_timestamp is None: print("警告: 无法从输入数据中确定最后一个已知时间戳。")


        df_processed_features = self._preprocess_input_for_prediction(df_raw.copy()) 
        if len(df_processed_features) < self.model_config['look_back']:
            raise ValueError(f"处理后的预测数据长度不足 ({len(df_processed_features)})。需要至少 {self.model_config['look_back']} 条记录。")
        
        model_input_feature_names = self.model_config['all_feature_columns_for_sequence']
        # 确保只选择存在的列，以避免KeyError
        last_input_sequence_df = df_processed_features[[col for col in model_input_feature_names if col in df_processed_features.columns]].iloc[-self.model_config['look_back']:]
        
        X_pred_np = np.array([last_input_sequence_df.values])
        if X_pred_np.size == 0: print("无法创建预测序列。"); return None, None
        X_pred_torch = torch.from_numpy(X_pred_np).float().to(DEVICE)
        with torch.no_grad(): predictions_scaled = self.model(X_pred_torch)
        predictions_original_all_targets = np.zeros_like(predictions_scaled.cpu().numpy())
        for i, col_name in enumerate(self.model_config['target_col_names']):
            scaler = self.target_scalers[col_name]
            pred_col_scaled = predictions_scaled.cpu().numpy()[0, :, i].reshape(-1, 1)
            predictions_original_all_targets[0, :, i] = scaler.inverse_transform(pred_col_scaled).flatten()
        print(f"成功生成预测。"); return predictions_original_all_targets[0], last_known_timestamp

class AQISystem:
    """AQI预测与异常检测系统主类。"""
    def __init__(self, artifacts_dir=MODEL_ARTIFACTS_DIR, config_overrides=None):
        self.artifacts_dir = artifacts_dir
        self.config = self._load_default_config()
        if config_overrides:
            self.config.update(config_overrides)
        
        self.config['model_artifacts_dir'] = self.artifacts_dir # 确保配置中的路径与实例一致
        os.makedirs(self.artifacts_dir, exist_ok=True)
        set_seed()

        self.trainer = None 
        self.predictor_instance = None # 用于存储ModelPredictor的实例
        
        self.model_config = None
        self.feature_scaler = None
        self.target_scalers = None
        self.model = None
        self.all_feature_columns_for_sequence = None # 从模型配置中加载

    def _load_default_config(self):
        return {
            'file_path': DEFAULT_FILE_PATH, 'look_back': DEFAULT_LOOK_BACK, 'horizon': DEFAULT_HORIZON,
            'target_col_names': DEFAULT_TARGET_COL_NAMES, 'primary_target_col_name': DEFAULT_PRIMARY_TARGET_COL_NAME,
            'batch_size': DEFAULT_BATCH_SIZE, 'model_artifacts_dir': self.artifacts_dir, 
            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS, 'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS, 'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA, 'anomaly_threshold_factor': DEFAULT_ANOMALY_THRESHOLD_FACTOR
        }

    def _ensure_model_loaded_for_use(self):
        """确保模型、缩放器和配置已加载，供预测或异常检测使用。"""
        if self.model is not None and self.model_config is not None and \
           self.feature_scaler is not None and self.target_scalers is not None and \
           self.all_feature_columns_for_sequence is not None:
            return # 已加载

        print(f"从 {self.artifacts_dir} 加载模型及相关组件 (AQISystem)...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)

        if not all(os.path.exists(p) for p in [config_path, model_path, fs_path, ts_path]):
            missing = [p for p in [config_path, model_path, fs_path, ts_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"一个或多个必要的模型文件在 '{self.artifacts_dir}' 中未找到: {missing}。")

        with open(config_path, 'r', encoding='utf-8') as f: self.model_config = json.load(f)
        
        # 更新实例配置以匹配加载的模型配置
        self.config['look_back'] = self.model_config.get('look_back', self.config['look_back'])
        self.config['horizon'] = self.model_config.get('horizon', self.config['horizon'])
        self.config['target_col_names'] = self.model_config.get('target_col_names', self.config['target_col_names'])
        self.config['primary_target_col_name'] = self.model_config.get('primary_target_col_name', self.config['primary_target_col_name'])
        self.all_feature_columns_for_sequence = self.model_config.get('all_feature_columns_for_sequence')
        if self.all_feature_columns_for_sequence is None:
            raise ValueError("模型配置中未找到 'all_feature_columns_for_sequence'。")


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
        print("模型及相关组件加载成功 (AQISystem)。")

    def train_new_model(self, train_data_path):
        self.config['file_path'] = train_data_path
        self.trainer = ModelTrainer(self.config) 
        try:
            self.trainer.run_training_pipeline()
            # 训练后，主动加载新训练的组件到 AQISystem 实例
            self._ensure_model_loaded_for_use() 
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            import traceback; traceback.print_exc()

    def predict_with_existing_model(self, input_data_path_or_df):
        self._ensure_model_loaded_for_use() # 确保模型等已加载到 AQISystem

        # 使用 AQISystem 内部加载的组件，而不是重新实例化 ModelPredictor
        # ModelPredictor 的 predict 和 _preprocess_input_for_prediction 逻辑需要在这里复用或调用
        # 为了简化，暂时仍实例化 ModelPredictor，但理想情况下应避免重复加载
        if self.predictor_instance is None or self.predictor_instance.artifacts_dir != self.artifacts_dir:
             self.predictor_instance = ModelPredictor(self.artifacts_dir)
        
        # 确保 predictor_instance 使用的是 AQISystem 当前加载的 model 和 scalers
        # (这一步在当前 ModelPredictor 实现中是自动的，因为它在 __init__ 中加载)
        # 如果要优化，需要修改 ModelPredictor 使其可以接收已加载的组件

        predicted_values, last_timestamp = self.predictor_instance.predict(input_data_path_or_df)

        if predicted_values is not None: # last_timestamp 可能为 None
            print(f"\n预测的 {self.predictor_instance.model_config['horizon']} 小时指标值 (原始尺度):")
            for h in range(min(5, self.predictor_instance.model_config['horizon'])): # 显示前5个小时或所有（如果不足5个）
                hour_str = f"  小时 {h+1}: "; 
                for t_idx, t_name in enumerate(self.predictor_instance.model_config['target_col_names']): 
                    val_to_print = predicted_values[h, t_idx]
                    if t_name == 'CO': hour_str += f"{t_name}={val_to_print:.2f} "
                    else: hour_str += f"{t_name}={np.round(val_to_print).astype(int)} "
                print(hour_str)
            
            if last_timestamp is not None: # 只有在有时间戳时才提供保存选项
                save_pred = input("是否将预测结果保存到CSV文件? (y/n): ").strip().lower() 
                if save_pred.startswith('y'): 
                    pred_save_path = os.path.join(self.artifacts_dir, "predictions_output_adv.csv") 
                    future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=self.predictor_instance.model_config['horizon'], freq='H')
                    output_data = {'date': future_timestamps.strftime('%Y%m%d'), 'hour': future_timestamps.hour}
                    
                    for t_idx, t_name in enumerate(self.predictor_instance.model_config['target_col_names']): 
                        val_col = predicted_values[:, t_idx]
                        if t_name == 'CO': output_data[t_name] = np.round(val_col, 2)
                        else: output_data[t_name] = np.round(val_col).astype(int)
                            
                    requested_output_columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
                    final_output_df_cols = {}
                    for col in requested_output_columns:
                        final_output_df_cols[col] = output_data.get(col, [np.nan] * self.predictor_instance.model_config['horizon'])
                    output_df = pd.DataFrame(final_output_df_cols)[requested_output_columns] # 保证列顺序
                    
                    output_df.to_csv(pred_save_path, index=False, encoding='utf-8-sig') 
                    print(f"预测结果已保存到: {pred_save_path}")
                    # (与原脚本类似的额外打印信息)
                else: print("预测结果未保存。")
            else:
                print("由于无法确定最后一个时间戳，预测结果未链接到具体日期，也无法保存为带时间戳的CSV。")

        elif predicted_values is None: print("未能生成预测。")


    def _preprocess_input_for_anomaly(self, df_raw):
        """为异常检测预处理输入数据。返回包含缩放后特征和原始尺度目标的DataFrame。"""
        print("为异常检测预处理输入数据 (AQISystem)...")
        df_processed = df_raw.copy()

        # 1. 时间戳处理
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
                except Exception as e: raise ValueError(f"无法为异常检测数据设置时间索引: {e}。")

        # 2. 转换为数值类型，并保留原始目标值副本（如果需要，但create_sequences会从df_processed取）
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 3. 处理NaN：特征列用ffill/bfill/0，目标列用0填充（异常检测需要实际值，NaN表示数据缺失）
        feature_cols_in_df_anomaly = [col for col in df_processed.columns if col not in self.model_config['target_col_names']]
        for col in feature_cols_in_df_anomaly:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        for tc in self.model_config['target_col_names']:
            if tc in df_processed.columns:
                if df_processed[tc].isnull().any():
                    print(f"警告: 目标列 '{tc}' 在异常检测数据中包含NaN值。将用0填充。")
                    df_processed[tc] = df_processed[tc].fillna(0)
            else: # 如果目标列完全缺失
                print(f"警告: 异常检测数据缺少目标列 '{tc}'。将创建为全0列。")
                df_processed[tc] = 0 # 创建全0列

        # 4. 特征工程 (AQI_calculated, cyclical, lags)
        if 'AQI_calculated' in self.all_feature_columns_for_sequence: # 使用 AQISystem 的属性
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi:
                df_processed = calculate_aqi_from_pollutants(df_processed.copy())
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                     df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            else:
                print("警告: 异常检测数据缺少计算'AQI_calculated'的列，用0填充。")
                df_processed['AQI_calculated'] = 0
        
        new_cyclical_features = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            idx = df_processed.index
            new_cyclical_features['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
        df_processed = pd.concat([df_processed, new_cyclical_features], axis=1)

        lag_features_to_recreate = [f for f in self.all_feature_columns_for_sequence if "_lag_" in f]
        lag_series_list = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_")
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]; lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns:
                    lag_series_list.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        if lag_series_list: df_processed = pd.concat([df_processed] + lag_series_list, axis=1)

        # 再次填充NaN (滞后特征可能引入NaN)
        for col in df_processed.columns:
            if df_processed[col].isnull().any(): df_processed[col] = df_processed[col].ffill().bfill().fillna(0)

        # 5. 准备特征进行缩放，并组合最终DataFrame
        df_features_for_scaling = pd.DataFrame(index=df_processed.index)
        for f_col in self.all_feature_columns_for_sequence: # 使用 AQISystem 的属性
            if f_col not in df_processed.columns:
                print(f"警告: 异常检测数据中缺少特征 '{f_col}' (工程化后)。用0填充。")
                df_features_for_scaling[f_col] = 0
            else:
                df_features_for_scaling[f_col] = df_processed[f_col]
        
        for f_col in self.all_feature_columns_for_sequence: # 确保数值类型
             df_features_for_scaling[f_col] = pd.to_numeric(df_features_for_scaling[f_col], errors='coerce').fillna(0)
             if not pd.api.types.is_numeric_dtype(df_features_for_scaling[f_col]):
                try: df_features_for_scaling[f_col] = df_features_for_scaling[f_col].astype(float)
                except ValueError: raise ValueError(f"无法将异常检测特征列 '{f_col}' 转为浮点数。")
        
        # 6. 缩放特征
        df_scaled_features_part = self.feature_scaler.transform(df_features_for_scaling[self.all_feature_columns_for_sequence])
        df_scaled_features_df = pd.DataFrame(df_scaled_features_part, columns=self.all_feature_columns_for_sequence, index=df_processed.index)

        # 7. 组合缩放后的特征和原始尺度的目标列
        df_final_for_sequences = df_scaled_features_df.copy()
        for tc in self.model_config['target_col_names']:
            if tc in df_processed.columns: # 应该是True
                df_final_for_sequences[tc] = df_processed[tc] # 使用清理后的原始尺度目标值
            else: # 不应该发生
                 df_final_for_sequences[tc] = 0 
        
        # 最后清理一次NaN，确保输入到create_sequences的数据是干净的
        cols_for_sequence_creation = self.all_feature_columns_for_sequence + self.model_config['target_col_names']
        for col in cols_for_sequence_creation:
            if col in df_final_for_sequences.columns and df_final_for_sequences[col].isnull().any():
                print(f"警告: 列 '{col}' 在最终序列创建前仍有NaN，用0填充。")
                df_final_for_sequences[col] = df_final_for_sequences[col].fillna(0)
        
        if df_final_for_sequences.empty:
            raise ValueError("数据在为异常检测预处理后变为空。")
        return df_final_for_sequences


    def detect_anomalies(self, data_path_or_df, threshold_factor=None):
        self._ensure_model_loaded_for_use()

        current_threshold_factor = threshold_factor if threshold_factor is not None else self.config.get('anomaly_threshold_factor', DEFAULT_ANOMALY_THRESHOLD_FACTOR)
        print(f"开始使用阈值因子 {current_threshold_factor} 进行异常检测...")

        if isinstance(data_path_or_df, str):
            try: df_raw_full = pd.read_csv(data_path_or_df)
            except Exception: df_raw_full = pd.read_excel(data_path_or_df)
        elif isinstance(data_path_or_df, pd.DataFrame): df_raw_full = data_path_or_df.copy()
        else: raise ValueError("输入数据必须是文件路径(CSV/Excel)或Pandas DataFrame。")

        df_for_sequences = self._preprocess_input_for_anomaly(df_raw_full.copy())

        # 确保所有目标列都存在于原始数据中，以便进行比较
        missing_targets = [tc for tc in self.model_config['target_col_names'] if tc not in df_for_sequences.columns or df_for_sequences[tc].isnull().all()]
        if missing_targets:
            raise ValueError(f"异常检测数据预处理后缺少有效的目标列: {missing_targets}。")

        X_anomaly_np, y_actual_anomaly_np = create_sequences(
            df_for_sequences, 
            look_back=self.model_config['look_back'],
            horizon=1, # 我们比较的是模型对下一步的预测
            target_col_names=self.model_config['target_col_names'], 
            feature_cols=self.all_feature_columns_for_sequence, # 使用 AQISystem 的属性
            is_predict=False
        )

        if X_anomaly_np.size == 0:
            print("无法为异常检测创建序列。数据可能太短或预处理后为空。")
            return {} # 返回空字典表示没有检测结果

        # 获取模型预测
        self.model.eval()
        all_predictions_scaled_list = []
        # 分批处理以防数据过大
        anomaly_dataset = TensorDataset(torch.from_numpy(X_anomaly_np).float())
        # 使用配置中的batch_size，如果未定义，则使用默认值
        batch_size_for_anomaly = self.config.get('batch_size', DEFAULT_BATCH_SIZE) 
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size_for_anomaly, shuffle=False)
        
        with torch.no_grad():
            for x_batch_tuple in anomaly_loader:
                x_batch = x_batch_tuple[0].to(DEVICE) # DataLoader返回元组
                outputs = self.model(x_batch) # 模型预测完整的horizon
                all_predictions_scaled_list.append(outputs.cpu().numpy())
        
        if not all_predictions_scaled_list:
            print("模型未能对异常检测数据生成预测。")
            return {}
            
        predictions_scaled_full_horizon = np.concatenate(all_predictions_scaled_list, axis=0)
        predictions_scaled_one_step = predictions_scaled_full_horizon[:, 0, :] # (num_sequences, num_targets)

        # 反向转换预测值和实际值 (实际值y_actual_anomaly_np已经是原始尺度)
        predicted_original_one_step = np.zeros_like(predictions_scaled_one_step)
        for i, col_name in enumerate(self.model_config['target_col_names']):
            scaler = self.target_scalers[col_name]
            predicted_original_one_step[:, i] = scaler.inverse_transform(predictions_scaled_one_step[:, i].reshape(-1,1)).flatten()
        
        # y_actual_anomaly_np 已经是原始尺度，形状 (num_sequences, 1, num_targets)
        actual_original_one_step = y_actual_anomaly_np[:, 0, :]

        # 计算误差
        errors = np.abs(actual_original_one_step - predicted_original_one_step) # MAE

        anomaly_reports = {}
        print("\n--- 异常检测报告 ---")
        num_sequences_generated = X_anomaly_np.shape[0]
        # 异常点的时间戳应对应于 y_actual_anomaly_np 的时间戳
        # y_actual_anomaly_np[k] 对应于原始数据中 df_raw_full.index[look_back + k] 的目标值
        # (假设 df_for_sequences 在 create_sequences 前未因dropna等操作大幅改变索引对应关系)
        # 更稳妥的方式是使用 df_for_sequences 在 create_sequences 之前的索引
        
        # 获取用于绘图和报告的时间戳/索引
        # 这些时间戳对应于 `y_actual_anomaly_np` 中的每个点
        # `df_for_sequences` 是传递给 `create_sequences` 的数据
        # `y_actual_anomaly_np[k]` 的时间戳是 `df_for_sequences.index[self.model_config['look_back'] + k]`
        
        # 确保 df_for_sequences 的索引用作时间戳基础
        base_timestamps_for_y = df_for_sequences.index[self.model_config['look_back'] : self.model_config['look_back'] + num_sequences_generated]


        for i, col_name in enumerate(self.model_config['target_col_names']):
            col_errors = errors[:, i]
            if len(col_errors) == 0: # 如果某个目标没有误差数据（不太可能发生，除非前面处理有误）
                print(f"\n目标: {col_name} - 无误差数据可供分析。")
                anomaly_reports[col_name] = {'count': 0, 'threshold': np.nan, 'timestamps': []}
                continue

            mean_error = np.mean(col_errors); std_error = np.std(col_errors)
            # 防止std_error为0（例如，如果所有误差都相同）
            threshold = mean_error + current_threshold_factor * std_error if std_error > 1e-9 else mean_error + 1e-9 
            
            anomaly_flags_col = col_errors > threshold
            current_col_anomaly_indices_relative = np.where(anomaly_flags_col)[0] # 相对于 errors 数组的索引

            # 将相对索引映射回原始数据的时间戳
            anomaly_timestamps_col = [base_timestamps_for_y[k] for k in current_col_anomaly_indices_relative]

            anomaly_reports[col_name] = {
                'count': len(anomaly_timestamps_col),
                'threshold': threshold, 'mean_error': mean_error, 'std_error': std_error,
                'indices_relative_to_error_array': current_col_anomaly_indices_relative.tolist(),
                'timestamps': anomaly_timestamps_col
            }
            print(f"\n目标: {col_name}")
            print(f"  计算阈值: {threshold:.4f} (基于 平均误差 {mean_error:.4f} + {current_threshold_factor} * 标准差 {std_error:.4f})")
            if len(anomaly_timestamps_col) > 0:
                print(f"  检测到 {len(anomaly_timestamps_col)} 个异常点:")
                for ts_idx, ts in enumerate(anomaly_timestamps_col[:5]): # 打印前5个
                    actual_val = actual_original_one_step[current_col_anomaly_indices_relative[ts_idx], i]
                    predicted_val = predicted_original_one_step[current_col_anomaly_indices_relative[ts_idx], i]
                    error_val = col_errors[current_col_anomaly_indices_relative[ts_idx]]
                    print(f"    - 时间戳: {ts}, 实际值: {actual_val:.2f}, 预测值: {predicted_val:.2f}, 误差: {error_val:.2f}")
                if len(anomaly_timestamps_col) > 5: print(f"    ...等 (共 {len(anomaly_timestamps_col)} 个)")
            else: print("  未检测到异常点。")
        
            # 绘图
            plot_save_prefix = os.path.join(self.artifacts_dir, f"anomaly_detection") # 统一前缀
            plot_anomalies(
                timestamps_or_indices=base_timestamps_for_y, # 使用 y 对应的时间戳
                actual_values=actual_original_one_step[:, i], # 对应 y 的实际值
                anomaly_indices=current_col_anomaly_indices_relative, # 异常点的相对索引
                target_name=col_name,
                save_path_prefix=f"{plot_save_prefix}_{col_name}", # 为每个目标单独文件
                title_suffix=f"异常点 (因子={current_threshold_factor})"
            )
            print(f"异常点图表已保存: {plot_save_prefix}_{col_name}_anomalies.png")
        return anomaly_reports


if __name__ == "__main__":
    print(f"使用设备: {DEVICE}")
    
    action = input("您想做什么？ (1: 训练新模型, 2: 使用现有模型进行预测, 3: 使用现有模型检测异常): ").strip()

    if action == '1':
        custom_artifacts_dir = input(f"请输入模型文件保存目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        train_data_path = input(f"请输入训练数据文件路径 (默认为 '{DEFAULT_FILE_PATH}'): ").strip() or DEFAULT_FILE_PATH
        if not os.path.exists(train_data_path):
            print(f"错误: 训练数据文件 '{train_data_path}' 未找到。")
        else:
            system = AQISystem(artifacts_dir=custom_artifacts_dir)
            system.train_new_model(train_data_path=train_data_path)

    elif action == '2':
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        input_data_source = input("请输入用于预测的数据文件路径 (CSV 或 Excel): ").strip()
        if not os.path.exists(input_data_source):
            print(f"错误: 预测数据文件 '{input_data_source}' 未找到。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir)
                system.predict_with_existing_model(input_data_path_or_df=input_data_source)
            except FileNotFoundError as e: print(f"文件未找到错误: {e}。请检查路径和文件名。")
            except ValueError as e: print(f"预测过程中发生值错误: {e}")
            except Exception as e: print(f"预测过程中发生未知错误: {e}"); import traceback; traceback.print_exc()

    elif action == '3':
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        anomaly_data_path = input("请输入用于异常检测的数据文件路径 (CSV 或 Excel，应包含实际值): ").strip()
        threshold_factor_str = input(f"请输入异常检测的阈值因子 (例如 3.0，默认为 {DEFAULT_ANOMALY_THRESHOLD_FACTOR}): ").strip()
        
        custom_threshold_factor = None
        if threshold_factor_str:
            try: custom_threshold_factor = float(threshold_factor_str)
            except ValueError: print("无效的阈值因子，将使用默认值。")

        if not os.path.exists(anomaly_data_path):
            print(f"错误: 异常检测数据文件 '{anomaly_data_path}' 未找到。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir)
                system.detect_anomalies(data_path_or_df=anomaly_data_path, threshold_factor=custom_threshold_factor)
            except FileNotFoundError as e: print(f"文件未找到错误: {e}。请检查路径和文件名。")
            except ValueError as e: print(f"异常检测过程中发生值错误: {e}")
            except Exception as e: print(f"异常检测过程中发生未知错误: {e}"); import traceback; traceback.print_exc()
    else:
        print("无效的选择。请输入 '1', '2', 或 '3'。")

    print("\nAQI系统流程结束。")
