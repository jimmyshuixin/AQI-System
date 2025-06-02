# ==============================================================================
# 主要功能：
# 本脚本实现了一个基于Transformer模型的多目标空气质量指标（AQI及多种污染物浓度）预测系统。
# 功能包括：
# 1. 数据加载与高级预处理：
#    - 时间戳处理与索引设置。
#    - 稳健的数值转换和缺失值（NaN）填充策略，特别处理新增的周期性特征列。
#    - 基于IQR的异常值检测与插值修复。
#    - 特征工程：
#        - 根据中国环保标准（GB 3095-2012）的污染物浓度限值计算AQI（作为特征"AQI_calculated"）。
#        - 创建时间周期性特征（如小时、星期、月份的正余弦编码）。
#        - 为目标变量和关键的周期性特征（如 *_24h, *_8h 列）创建滞后特征。
# 2. 模型训练：
#    - 使用PyTorch实现的Transformer模型架构。
#    - 支持多目标同时预测。
#    - 采用Optuna进行超参数优化，搜索最佳模型配置。
#    - 训练过程中使用学习率调度器（ReduceLROnPlateau）和早停机制。
# 3. 模型评估：
#    - 在测试集上评估模型性能，计算MAE, RMSE, R²等指标（针对每个目标污染物）。
#    - 可视化预测结果与实际值的对比。
# 4. 模型预测：
#    - 加载已训练的模型和相关组件（缩放器、配置）。
#    - 对新的输入数据进行预测。
#    - 预测结果可以保存为CSV文件，并根据要求对特定列（如CO）进行小数位数格式化。
# 5. 用户交互：
#    - 脚本运行时会询问用户是进行模型训练还是加载模型进行预测。
#
# 设计思路与优化：
# - 模块化设计：通过ModelTrainer和ModelPredictor类封装训练和预测逻辑。
# - 鲁棒性：增强了数据预处理的鲁棒性，特别是对NaN和异常值的处理。
# - 性能：优化了特征创建过程，减少DataFrame碎片化。
# - 准确度：通过广泛的超参数搜索、特征工程和学习率调度等手段，力求最大化预测准确度。
# - 可读性：提供详尽的中文注释，便于理解和维护。
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
# 尝试设置matplotlib以支持中文显示，如果系统支持SimHei字体。
# axes.unicode_minus = False 用于正常显示负号。
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"设置中文字体失败: {e}。绘图标签可能无法正确显示中文。")

# --- 全局参数设置 ---
DEFAULT_FILE_PATH = '南京_AQI_Data.xlsx' # 默认的训练数据文件路径
DEFAULT_LOOK_BACK = 24 # 模型回溯（参考历史数据）的时间步长，单位：小时
DEFAULT_HORIZON = 72   # 模型预测未来的时间步长，单位：小时
# 定义模型需要预测的目标污染物列表
DEFAULT_TARGET_COL_NAMES = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
# 数据集中所有可能用到的列名，用于预处理时参考
DEFAULT_ALL_AVAILABLE_COL_NAMES = [ 
    'date', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
    'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 
    'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'
]
# 在Optuna优化和早停机制中，主要参考的目标污染物（通常是AQI）
DEFAULT_PRIMARY_TARGET_COL_NAME = 'AQI' 

DEFAULT_BATCH_SIZE = 32 # 训练和评估时每个批次的数据量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择GPU或CPU

# --- 文件保存路径 ---
MODEL_ARTIFACTS_DIR = "model_artifacts_advanced" # 保存所有模型相关文件的目录名
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model_adv.pth" # 保存最佳模型状态的文件名
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler_adv.pkl" # 保存特征缩放器的文件名
TARGET_SCALERS_SAVE_NAME = "aqi_target_scalers_adv.pkl" # 保存目标缩放器字典的文件名
MODEL_CONFIG_SAVE_NAME = "model_config_adv.json" # 保存模型配置（架构参数、列名等）的文件名

# --- 训练特定参数 (为最大化准确度，参数设置较为激进) ---
DEFAULT_FULL_TRAIN_EPOCHS = 200  # 最终模型训练的总轮数
DEFAULT_N_OPTUNA_TRIALS = 150    # Optuna超参数优化的试验次数
DEFAULT_OPTUNA_EPOCHS = 30     # Optuna单次试验的训练轮数
DEFAULT_EARLY_STOPPING_PATIENCE = 20 # 早停机制的耐心轮数（连续多少轮验证集性能无提升则停止）
DEFAULT_MIN_DELTA = 0.00001 # 判断验证集性能是否有“显著”提升的最小阈值

# --- AQI 计算相关的常量 ---
# IAQI（个体空气质量指数）的等级划分标准
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
# 各污染物浓度对应的IAQI计算断点值 (µg/m³，CO为mg/m³)
# 这些断点值基于中国的环境空气质量标准 (GB 3095-2012)，并对应特定的平均周期。
POLLUTANT_BREAKPOINTS = {
    'SO2_24h':   [0, 50,  150,  475,  800,  1600, 2100, 2620], # 二氧化硫 24小时平均浓度限值
    'NO2_24h':   [0, 40,  80,   180,  280,  565,  750,  940],  # 二氧化氮 24小时平均浓度限值
    'PM10_24h':  [0, 50,  150,  250,  350,  420,  500,  600],  # PM10 24小时平均浓度限值
    'CO_24h':    [0, 2,   4,    14,   24,   36,   48,   60],   # 一氧化碳 24小时平均浓度限值 (mg/m³)
    'O3_8h_24h': [0, 100, 160,  215,  265,  800, 1000, 1200],  # 臭氧 日最大8小时滑动平均浓度限值
    'O3_1h':     [0, 160, 200,  300,  400,  800, 1000, 1200],  # 臭氧 1小时平均浓度限值 (备用或特殊情况)
    'PM2.5_24h': [0, 35,  75,   115,  150,  250,  350,  500]   # PM2.5 24小时平均浓度限值
}
# 为原始污染物（通常是小时值）定义一个回退的断点表，以防特定周期的平均值不可用。
# 注意：这些小时值断点可能与官方标准中的小时限值有所不同，仅作为近似或回退方案。
POLLUTANT_BREAKPOINTS_HOURLY_APPROX = {
    'SO2':   [0, 150, 500,  650,  800], # 二氧化硫 1小时平均 (µg/m³) - 示例，实际需查阅小时标准
    'NO2':   [0, 100, 200,  700, 1200], # 二氧化氮 1小时平均 (µg/m³) - 示例
    'PM10':  [0, 50, 150, 250, 350, 420], # PM10 1小时 (官方标准中可能无直接小时IAQI，常参考24h均值)
    'CO':    [0, 5,  10,   35,   60],    # 一氧化碳 1小时平均 (mg/m³) - 示例
    'O3':    POLLUTANT_BREAKPOINTS['O3_1h'], # 臭氧 1小时平均 (µg/m³)
    'PM2.5': [0, 35, 75, 115, 150, 250] # PM2.5 1小时 (官方标准中可能无直接小时IAQI，常参考24h均值)
}


# --- 工具函数 ---
def set_seed(seed_value=42):
    """设置全局随机种子以保证实验结果的可复现性。"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # 确保CUDA操作的确定性
        torch.backends.cudnn.benchmark = False # 禁用基准测试模式以保证确定性
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def calculate_iaqi(Cp, pollutant_key):
    """
    计算单个污染物的个体空气质量指数 (IAQI)。
    参数:
        Cp (float): 污染物浓度值。
        pollutant_key (str): 污染物类型键名，用于查找对应的浓度限值表 (如 'PM2.5_24h', 'O3_1h')。
    返回:
        float: 计算得到的IAQI值，如果无法计算则返回np.nan。
    """
    if pd.isna(Cp) or Cp < 0: # 无效浓度值处理
        return np.nan 

    # 选择合适的浓度限值表
    bp_table_to_use = POLLUTANT_BREAKPOINTS 
    if pollutant_key not in bp_table_to_use:
        bp_table_to_use = POLLUTANT_BREAKPOINTS_HOURLY_APPROX # 回退到小时近似表
        if pollutant_key not in bp_table_to_use:
            # print(f"警告: 未找到污染物 '{pollutant_key}' 的浓度限值表。无法计算其IAQI。") # 日志可能过多，可注释
            return np.nan
            
    bp = bp_table_to_use.get(pollutant_key) # 获取断点列表

    # IAQI计算逻辑
    if Cp > bp[-1]: # 浓度超过最高限值，IAQI封顶为500 (或根据标准进行外插)
        return 500 
    for i in range(len(bp) - 1):
        if bp[i] <= Cp < bp[i+1]: # 找到浓度所在的区间
            IAQI_Lo, IAQI_Hi = IAQI_LEVELS[i], IAQI_LEVELS[i+1]
            BP_Lo, BP_Hi = bp[i], bp[i+1]
            if BP_Hi == BP_Lo: return IAQI_Lo # 防止除以零
            # 线性插值计算IAQI
            return round(((IAQI_Hi - IAQI_Lo) / (BP_Hi - BP_Lo)) * (Cp - BP_Lo) + IAQI_Lo)
    if Cp == bp[0]: # 浓度等于最低限值
        return IAQI_LEVELS[0]
    return np.nan # 未落入任何区间 (理论上不应发生)

def calculate_aqi_from_pollutants(df):
    """
    根据输入的污染物浓度DataFrame计算AQI，并确定首要污染物。
    优先使用数据中提供的特定周期列（如 PM2.5_24h）进行IAQI计算。
    参数:
        df (pd.DataFrame): 包含污染物浓度数据的DataFrame。
    返回:
        pd.DataFrame: 增加了 'AQI_calculated' 和 'Primary_Pollutant_calculated' 列的DataFrame。
    """
    iaqi_df = pd.DataFrame(index=df.index) # 用于存储各污染物IAQI的临时DataFrame

    # 定义用于计算AQI的污染物及其在输入DataFrame中的首选和次选列名
    # 格式: {标准IAQI计算键名: [DataFrame中的首选列, DataFrame中的次选列, ...]}
    pollutants_for_calc = {
        'SO2_24h':   ['SO2_24h', 'SO2'], 
        'NO2_24h':   ['NO2_24h', 'NO2'],
        'PM10_24h':  ['PM10_24h', 'PM10'], 
        'CO_24h':    ['CO_24h', 'CO'],
        'O3_8h_24h': ['O3_8h_24h', 'O3_8h', 'O3'], # O3的优先级：日最大8h -> 8h滑动 -> 1h
        'PM2.5_24h': ['PM2.5_24h', 'PM2.5']
    }

    # 为每种标准污染物计算IAQI
    for bp_key, df_col_options in pollutants_for_calc.items():
        selected_col_for_iaqi = None
        for df_col in df_col_options: # 遍历列选项，找到第一个可用的数据列
            if df_col in df.columns and not df[df_col].isnull().all():
                selected_col_for_iaqi = df_col
                break 
        
        if selected_col_for_iaqi:
            # 使用bp_key（如'PM2.5_24h'）查找断点表，使用selected_col_for_iaqi（如'PM2.5_24h'或'PM2.5'）获取数据
            iaqi_df[bp_key] = df[selected_col_for_iaqi].apply(lambda x: calculate_iaqi(x, bp_key if bp_key in POLLUTANT_BREAKPOINTS else selected_col_for_iaqi))
        else:
            iaqi_df[bp_key] = np.nan # 如果所有选项都不可用，则该污染物的IAQI为NaN
            
    # AQI是所有污染物IAQI中的最大值
    df['AQI_calculated'] = iaqi_df.max(axis=1, skipna=True)
    
    # 确定首要污染物
    def get_primary_pollutants(row):
        if pd.isna(row['AQI_calculated']) or row['AQI_calculated'] <= 50: return '无' # AQI<=50时无首要污染物
        # 找到所有IAQI值等于最终AQI的污染物
        primary = [pollutant_bp_key for pollutant_bp_key in iaqi_df.columns 
                   if pd.notna(row[pollutant_bp_key]) and round(row[pollutant_bp_key]) == round(row['AQI_calculated'])]
        return ', '.join(primary) if primary else '无'

    # 为确定首要污染物，需要将计算出的AQI值临时加入iaqi_df
    temp_iaqi_df_for_primary = iaqi_df.copy()
    temp_iaqi_df_for_primary['AQI_calculated'] = df['AQI_calculated']
    df['Primary_Pollutant_calculated'] = temp_iaqi_df_for_primary.apply(get_primary_pollutants, axis=1)
    
    return df

def create_sequences(data_df, look_back, horizon, target_col_names, feature_cols, is_predict=False):
    """
    从时间序列数据创建输入序列 (X) 和输出序列 (y)。
    参数:
        data_df (pd.DataFrame): 包含特征和目标的时间序列数据。
        look_back (int): 输入序列的长度（回溯窗口）。
        horizon (int): 输出序列的长度（预测未来多少步）。
        target_col_names (list): 目标列的名称列表。
        feature_cols (list): 特征列的名称列表。
        is_predict (bool): 是否为预测模式。如果是，则不创建y。
    返回:
        tuple: (X_arr, y_arr) 或 X_arr (如果is_predict=True)。
               X_arr形状: (num_sequences, look_back, num_features)
               y_arr形状: (num_sequences, horizon, num_targets)
    """
    X_list, y_list = [], []
    # 检查特征列和目标列是否存在
    missing_feature_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_feature_cols: raise ValueError(f"数据缺少特征列: {missing_feature_cols}.")
    if not is_predict:
        missing_target_cols = [col for col in target_col_names if col not in data_df.columns]
        if missing_target_cols: raise ValueError(f"数据缺少目标列: {missing_target_cols}.")

    # 将DataFrame转换为NumPy数组以提高效率
    data_features_np = data_df[feature_cols].values
    if not is_predict: data_targets_np = data_df[target_col_names].values
    
    num_samples = len(data_features_np)
    # 计算可以生成的序列数量
    if is_predict: # 预测模式下，只需要X，不需要为y留出空间
        num_possible_sequences = num_samples - look_back + 1
    else: # 训练/评估模式下，需要为y的horizon留出空间
        num_possible_sequences = num_samples - look_back - horizon + 1

    if num_possible_sequences <= 0: # 如果数据太短，无法生成任何完整序列
        return np.array(X_list), np.array(y_list) # 返回空数组

    # 生成序列
    for i in range(num_possible_sequences):
        X_list.append(data_features_np[i : i + look_back])
        if not is_predict:
            y_list.append(data_targets_np[i + look_back : i + look_back + horizon, :])
            
    X_arr = np.array(X_list) if X_list else np.array([])
    if is_predict: return X_arr
    y_arr = np.array(y_list) if y_list else np.array([])
    
    # 如果X已创建但y为空（在非预测模式下），这表示数据不足以形成对应的y，应返回空
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
    plt.figure(figsize=(15, 7)); actual_flat = actual_values.flatten()
    plt.plot(timestamps_or_indices, actual_flat, label=f'实际{target_name}', alpha=0.7)
    valid_idx = np.array(anomaly_indices, dtype=int); valid_idx = valid_idx[valid_idx < len(actual_flat)]
    if len(valid_idx) > 0:
        ts_np = np.array(timestamps_or_indices); x = ts_np[valid_idx]; y = np.array(actual_flat)[valid_idx]
        plt.scatter(x, y, color='red', label='异常点', marker='o', s=50, zorder=5)
    plt.title(f'{target_name} - {title_suffix}'); plt.xlabel('时间'); plt.ylabel(target_name); plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_{target_name}_anomalies.png"); plt.close()

def detect_anomalies_iqr_and_impute(df, column_names, factor=1.5, interpolation_method='time'):
    """
    使用IQR方法检测指定列中的异常点，并进行插值填充。
    对于插值后仍存在的NaN（通常在序列的开始或结束），使用该列的中位数进行填充。
    参数:
        df (pd.DataFrame): 输入的DataFrame。
        column_names (list): 需要进行异常检测和填充的列名列表。
        factor (float): IQR的倍数，用于定义异常值边界。
        interpolation_method (str): 传递给 df.interpolate() 的插值方法。
    返回:
        pd.DataFrame: 处理了异常值的DataFrame副本。
    """
    df_cleaned = df.copy(); print("开始数据异常值检测和插值填充...")
    for col_name in column_names:
        if col_name in df_cleaned.columns:
            # 确保列是数值类型，以便进行分位数计算和插值
            if not pd.api.types.is_numeric_dtype(df_cleaned[col_name]):
                print(f"警告: 列 '{col_name}' 不是数值类型，跳过异常处理。")
                continue

            original_nan_count = df_cleaned[col_name].isna().sum()
            Q1 = df_cleaned[col_name].quantile(0.25); Q3 = df_cleaned[col_name].quantile(0.75)
            IQR = Q3 - Q1
            # 只有当IQR有效时才进行异常检测 (IQR可能为0，例如当列中大部分值相同时)
            if pd.notna(IQR) and IQR > 1e-6: # 增加一个小的阈值防止IQR过小
                lower_bound = Q1 - factor * IQR; upper_bound = Q3 + factor * IQR
                outlier_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
                num_outliers = outlier_mask.sum()
                if num_outliers > 0: 
                    print(f"列'{col_name}'检测到{num_outliers}个异常值。尝试插值..."); 
                    df_cleaned.loc[outlier_mask, col_name] = np.nan # 将异常值标记为NaN以便插值
            else:
                print(f"列'{col_name}'的IQR为0或无效，跳过基于IQR的异常值标记。")

            # 执行插值
            if isinstance(df_cleaned.index, pd.DatetimeIndex): # 优先使用时间插值
                try: df_cleaned[col_name] = df_cleaned[col_name].interpolate(method=interpolation_method, limit_direction='both')
                except Exception as e: print(f"列'{col_name}'时间插值失败:{e}。尝试线性插值。"); df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            else: # 如果不是时间索引，使用线性插值
                df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            
            # 填充插值后可能剩余的NaN (通常是序列开头/结尾)
            if df_cleaned[col_name].isna().sum() > 0 : 
                median_val = df_cleaned[col_name].median() # 使用当前列（可能已部分插值）的中位数
                fill_value = median_val if pd.notna(median_val) else 0 # 如果中位数也是NaN，则用0填充
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
    """
    基于Transformer Encoder的多目标时间序列预测模型。
    参数:
        num_features (int): 输入特征的数量。
        d_model (int): Transformer模型的内部维度 (embedding维度)。
        nhead (int): 多头注意力机制中的头数。
        num_encoder_layers (int): Transformer编码器的层数。
        dim_feedforward (int): 编码器中前馈网络的维度。
        dropout (float): Dropout比率。
        horizon (int): 预测的时间步长。
        num_target_features (int): 要预测的目标特征数量。
        norm_first (bool): 是否在Transformer层中先应用层归一化。
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, num_target_features, norm_first=True):
        super(AQITransformer, self).__init__(); self.d_model=d_model; self.horizon=horizon; self.num_target_features=num_target_features
        self.input_embedding = nn.Linear(num_features, d_model) # 输入嵌入层
        # Transformer编码器层定义
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation='gelu', batch_first=True, norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers) # 堆叠编码器层
        self.output_layer = nn.Linear(d_model, horizon * num_target_features) # 输出层，预测所有目标在整个horizon上的值
    
    def forward(self, src):
        # src 形状: (batch_size, seq_len, num_features)
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model) # 输入嵌入并缩放
        
        # 简化的正弦位置编码
        seq_len = src_embedded.size(1)
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        if self.d_model % 2 != 0: pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)] # 处理d_model为奇数的情况
        else: pe[:, 1::2] = torch.cos(position * div_term)
        
        src_pos_encoded = src_embedded + pe.unsqueeze(0) # 将位置编码加到嵌入向量上
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded) # 位置编码后的Dropout
        
        encoder_output = self.transformer_encoder(src_pos_encoded) # Transformer编码器处理，形状: (batch, seq_len, d_model)
        # 使用编码器输出序列的最后一个时间步的表示作为预测的输入
        prediction_input = encoder_output[:, -1, :] # 形状: (batch, d_model)
        
        output_flat = self.output_layer(prediction_input) # 输出层预测，形状: (batch, horizon * num_targets)
        # 将扁平化的输出重塑为 (batch, horizon, num_targets) 以匹配多目标多步预测的格式
        output = output_flat.view(output_flat.size(0), self.horizon, self.num_target_features)
        return output

# --- 模型训练器类 ---
# (ModelTrainer, ModelPredictor 和主执行流程中的注释将同样进行重写和润色)
# ... (其余代码与您提供的最新版本中的ModelTrainer, ModelPredictor和主执行流程类似，但注释会更新)

# --- 模型训练器类 ---
class ModelTrainer:
    """
    该类封装了模型训练的整个流程，包括数据加载、预处理、超参数优化、
    模型训练、评估以及相关文件的保存。
    """
    def __init__(self, config):
        """
        初始化ModelTrainer。
        参数:
            config (dict): 包含所有训练相关配置的字典。
        """
        self.config = config
        os.makedirs(self.config['model_artifacts_dir'], exist_ok=True) # 确保模型文件保存目录存在
        set_seed() # 设置随机种子以保证实验可复现
        self.all_feature_columns_for_sequence = [] # 初始化用于序列创建的特征列名列表
        self.feature_scaler = None # 初始化特征缩放器
        self.target_scalers = {}   # 初始化目标缩放器字典 (每个目标一个)

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True):
        """
        核心的数据加载和预处理函数。
        步骤包括：读取数据、时间戳处理、数值转换、缺失值填充、异常值处理（训练时）、
        特征工程（AQI计算、周期性特征、滞后特征）以及数据缩放。
        参数:
            file_path (str): 数据文件的路径。
            fit_scalers (bool): 是否拟合新的缩放器。训练时为True，（如果用于预测数据预处理则为False）。
        返回:
            pd.DataFrame: 经过预处理和缩放后的DataFrame。
        """
        print(f"开始加载和预处理数据从: {file_path}...")
        # 1. 读取数据 (尝试CSV和Excel)
        try: df = pd.read_csv(file_path)
        except Exception:
            try: df = pd.read_excel(file_path)
            except Exception as e_excel: print(f"读取Excel或CSV均失败: {e_excel}"); raise

        # 2. 时间戳处理和索引设置
        if 'date' in df.columns and 'hour' in df.columns: 
            # 根据 'date' 和 'hour' 列创建时间戳索引
            df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(int).astype(str).str.zfill(2), format='%Y%m%d%H')
            df = df.set_index('timestamp').drop(columns=['date', 'hour'], errors='ignore')
        elif 'Time' in df.columns: # 兼容旧的 'Time' 列名
             df['timestamp'] = pd.to_datetime(df['Time']); df = df.set_index('timestamp').drop(columns=['Time'], errors='ignore')
        else: # 尝试将第一列作为时间索引
            try: df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]); df = df.set_index(df.columns[0])
            except Exception: raise ValueError("无法自动设置时间索引。请确保数据包含 ('date' 和 'hour') 列，或 'Time' 列，或第一列是可解析的时间戳。")

        # 3. 将所有列尝试转换为数值类型，无法转换的设为NaN
        for col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 4. 处理非目标特征列中的NaN值 (前向填充 -> 后向填充 -> 0填充)
        feature_candidate_cols = [col for col in df.columns if col not in self.config['target_col_names']]
        for col in feature_candidate_cols: 
            if df[col].isnull().any():
                df[col] = df[col].ffill().bfill().fillna(0) 
        # 删除目标列中仍有NaN的行 (这些NaN可能是原始数据就有的，且无法通过特征填充解决)
        df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
        if df.empty: raise ValueError("数据在初步NaN处理后DataFrame为空（可能所有目标列都存在无法填充的NaN）。")

        # 5. 复杂异常处理 (仅在训练并拟合新缩放器时对目标列进行)
        if fit_scalers: 
            df = detect_anomalies_iqr_and_impute(df, self.config['target_col_names'])
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names']) # 清洗后再次检查
        if df.empty: raise ValueError("数据在异常值处理后DataFrame为空。")

        # 6. AQI 计算 (特征工程)
        df = calculate_aqi_from_pollutants(df) # 使用更新后的AQI计算函数
        # 填充计算后可能产生的NaN (例如，如果所有污染物浓度都缺失)
        if 'AQI_calculated' in df.columns and df['AQI_calculated'].isnull().any():
            df['AQI_calculated'] = df['AQI_calculated'].fillna(0)


        # 7. 创建周期性特征 (优化为使用pd.concat一次性添加)
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
        
        # 8. 创建滞后特征 (优化为使用pd.concat一次性添加)
        num_lags_to_create = max(1, self.config['look_back'] // 4) # 增加滞后特征数量
        lag_features_to_concat = [] # 存储待合并的滞后特征Series
        lag_cols_created_names = [] # 存储创建的滞后特征列名，用于后续dropna
        # 为目标列和所有以 _24h 或 _8h 结尾的列（通常是周期平均值）创建滞后特征
        cols_for_lags = self.config['target_col_names'] + \
                        [col for col in df.columns if (col.endswith('_24h') or col.endswith('_8h')) and col not in self.config['target_col_names']]
        cols_for_lags = sorted(list(set(cols_for_lags))) # 去重并排序，确保一致性

        for col_to_lag in cols_for_lags: 
            if col_to_lag in df.columns: # 确保原始列存在
                for lag in range(1, num_lags_to_create + 1):
                    lag_col_name = f"{col_to_lag}_lag_{lag}"
                    lag_features_to_concat.append(df[col_to_lag].shift(lag).rename(lag_col_name))
                    lag_cols_created_names.append(lag_col_name)
        
        if lag_features_to_concat: # 如果成功创建了任何滞后特征
            df = pd.concat([df] + lag_features_to_concat, axis=1)
            df = df.dropna(subset=lag_cols_created_names, how='any') # 删除因滞后产生的NaN行
        
        if df.empty: raise ValueError("数据在创建滞后特征后DataFrame为空。")

        # 9. 定义最终的特征列列表 (不包含原始目标列，但包含其滞后项、计算的AQI、周期特征等)
        self.all_feature_columns_for_sequence = [col for col in df.columns if col not in self.config['target_col_names']]
        # 特殊处理 AQI_calculated：如果它不是目标之一，则应作为特征
        if 'AQI_calculated' in df.columns and 'AQI_calculated' not in self.config['target_col_names']:
            if 'AQI_calculated' not in self.all_feature_columns_for_sequence:
                 self.all_feature_columns_for_sequence.append('AQI_calculated')
        elif 'AQI_calculated' in self.all_feature_columns_for_sequence and 'AQI_calculated' in self.config['target_col_names']:
             print("警告: 'AQI_calculated' 同时被定义为目标和特征。通常它应仅作为特征以避免数据泄露。")
        
        self.all_feature_columns_for_sequence = sorted(list(set(self.all_feature_columns_for_sequence))) # 去重并排序

        # 10. 检查目标列是否存在，准备缩放
        missing_targets_for_scaling = [tc for tc in self.config['target_col_names'] if tc not in df.columns]
        if missing_targets_for_scaling: raise ValueError(f"目标列在缩放前未找到: {missing_targets_for_scaling}.")
        
        # 11. 数据缩放 (仅在fit_scalers=True时拟合新的缩放器)
        if fit_scalers:
            self.feature_scaler = StandardScaler() # 或 RobustScaler 等
            # 确保用于缩放的特征列存在于DataFrame中
            current_features_to_scale = [f_col for f_col in self.all_feature_columns_for_sequence if f_col in df.columns] 
            
            # 在缩放前，对所有选定的特征列进行最终的数值类型检查和转换
            for f_col in current_features_to_scale: 
                if not pd.api.types.is_numeric_dtype(df[f_col]):
                    # print(f"特征列 '{f_col}' 在缩放前不是数值类型，尝试转换...") # 日志可能过多
                    df[f_col] = pd.to_numeric(df[f_col], errors='coerce')
                    if df[f_col].isnull().any(): 
                        # print(f"特征列 '{f_col}' 转换后包含NaN，用0填充。") # 日志可能过多
                        df[f_col] = df[f_col].fillna(0)
                    if not pd.api.types.is_numeric_dtype(df[f_col]): # 再次检查
                         # 如果仍然无法转换为数值，则抛出错误，指出问题列和一些非数值样本
                         offending_values = [item for item in df[f_col].unique() if not isinstance(item, (int, float, np.number))]
                         raise ValueError(f"特征列 '{f_col}' 无法转换为数值类型进行缩放。问题值示例: {offending_values[:5]}")
            
            if not current_features_to_scale: # 如果没有有效的特征列
                raise ValueError("没有有效的特征列可用于缩放。")

            df[current_features_to_scale] = self.feature_scaler.fit_transform(df[current_features_to_scale])
            joblib.dump(self.feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))
            
            self.target_scalers = {} # 为每个目标列创建并保存一个缩放器
            for col_name in self.config['target_col_names']:
                # 确保目标列也是数值类型
                if not pd.api.types.is_numeric_dtype(df[col_name]): 
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0) # 强制转为数值并填充
                scaler = StandardScaler(); df[[col_name]] = scaler.fit_transform(df[[col_name]])
                self.target_scalers[col_name] = scaler
            joblib.dump(self.target_scalers, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALERS_SAVE_NAME))
            print("特征和目标缩放器已拟合和保存。")
        return df

    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, trial=None):
        """核心模型训练循环，支持Optuna、早停和学习率调度。"""
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
        """Optuna的目标函数，用于超参数搜索。"""
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience) # 移除了 verbose=True
        criterion = nn.MSELoss() 
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=2, pin_memory=True) 
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
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
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
        final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_hyperparams['learning_rate'], weight_decay=best_hyperparams.get('weight_decay', 0.0))
        final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=best_hyperparams.get('scheduler_factor', 0.5), patience=best_hyperparams.get('scheduler_patience', 7)) # 移除了 verbose=True
        criterion = nn.MSELoss()
        final_model, train_losses, val_losses, _ = self._train_model_core(final_model, final_train_loader, final_val_loader, criterion, final_optimizer, final_scheduler, self.config['full_train_epochs'])
        plot_training_loss(train_losses, val_losses, os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"), title_prefix="最终模型 (Overall)")
        print(f"最终模型训练完成并保存。")
        model_config_to_save = {'model_architecture': final_model_arch_params, 'look_back': self.config['look_back'], 'horizon': self.config['horizon'],
            'target_col_names': self.config['target_col_names'], 'primary_target_col_name': self.config['primary_target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence, 'num_input_features_for_model': num_input_features_for_model,
            'num_target_features': len(self.config['target_col_names']), 'optuna_best_params': best_hyperparams } 
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w', encoding='utf-8') as f: json.dump(model_config_to_save, f, indent=4, ensure_ascii=False)
        print(f"模型配置已保存。")
        if X_test_np.shape[0] > 0: self.evaluate_trained_model(final_model, X_test_np, y_test_np, criterion)

    def evaluate_trained_model(self, model, X_test_np, y_test_np, criterion):
        """评估训练好的模型在测试集上的性能。"""
        print("\n评估最终模型在测试集上的性能...")
        test_loader = DataLoader(TimeSeriesDataset(X_test_np, y_test_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
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

class ModelPredictor:
    """封装加载已训练模型并进行预测的逻辑。"""
    def __init__(self, artifacts_dir):
        self.artifacts_dir = artifacts_dir; self.model = None; self.feature_scaler = None; self.target_scalers = {}; self.model_config = None
        self._load_artifacts()
    def _load_artifacts(self):
        print(f"从 {self.artifacts_dir} 加载模型及相关组件...")
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME); model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME); ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)
        if not all(os.path.exists(p) for p in [config_path, model_path, fs_path, ts_path]):
            missing_files = [p for p in [config_path, model_path, fs_path, ts_path] if not os.path.exists(p)]
            raise FileNotFoundError(f"一个或多个必要的模型文件在 '{self.artifacts_dir}' 中未找到: {missing_files}。")
        with open(config_path, 'r', encoding='utf-8') as f: self.model_config = json.load(f)
        self.feature_scaler = joblib.load(fs_path); self.target_scalers = joblib.load(ts_path)
        self.model = AQITransformer(num_features=self.model_config['num_input_features_for_model'], **self.model_config['model_architecture'], 
                                   horizon=self.model_config['horizon'], num_target_features=self.model_config['num_target_features']).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)); self.model.eval(); print("模型及相关组件加载成功。")

    def _preprocess_input_for_prediction(self, df_raw):
        """为预测预处理输入数据，与训练过程保持一致。"""
        print("预处理输入数据进行预测...")
        df_processed = df_raw.copy()
        if not isinstance(df_processed.index, pd.DatetimeIndex): 
            if 'Time' in df_processed.columns: df_processed['timestamp'] = pd.to_datetime(df_processed['Time']); df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                 # 修正: 确保 element-wise 操作
                 df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                 df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                 df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else:
                try:
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                        df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0]); df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: raise ValueError(f"无法为预测数据设置时间索引: {e}。")
        for col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 
        for col in df_processed.columns: 
            if df_processed[col].isnull().any(): df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
        can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
        if 'AQI_calculated' in self.model_config['all_feature_columns_for_sequence']:
            if can_calc_aqi: df_processed = calculate_aqi_from_pollutants(df_processed)
            else: print("警告: 预测数据缺少计算'AQI_calculated'的列，用0填充。"); df_processed['AQI_calculated'] = 0 
        
        new_cyclical_features_pred = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex): 
            idx = df_processed.index
            new_cyclical_features_pred['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features_pred['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features_pred['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features_pred['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features_pred['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features_pred['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
        df_processed = pd.concat([df_processed, new_cyclical_features_pred], axis=1)
        
        lag_features_to_recreate = [f for f in self.model_config['all_feature_columns_for_sequence'] if "_lag_" in f] 
        lag_series_list_pred = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_"); 
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]; lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns: 
                    lag_series_list_pred.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: # 如果原始列不存在，创建一个全0的Series
                    lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: 
                lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        
        if lag_series_list_pred:
             df_processed = pd.concat([df_processed] + lag_series_list_pred, axis=1)

        # 在合并滞后特征后，对所有列（特别是新创建的滞后列）进行填充
        for col in df_processed.columns:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        expected_features_from_config = self.model_config['all_feature_columns_for_sequence'] 
        for f_col in expected_features_from_config:
            if f_col not in df_processed.columns: print(f"警告: 预测数据中缺少特征 '{f_col}'。用0填充。"); df_processed[f_col] = 0
        
        df_for_scaling = df_processed[expected_features_from_config].copy()
        for f_col in expected_features_from_config: # 确保数值类型
            df_for_scaling[f_col] = pd.to_numeric(df_for_scaling[f_col], errors='coerce').fillna(0)
            if not pd.api.types.is_numeric_dtype(df_for_scaling[f_col]):
                try: df_for_scaling[f_col] = df_for_scaling[f_col].astype(float)
                except ValueError as e_astype:
                    offending_values = [item for item in df_for_scaling[f_col].unique() if not isinstance(item, (int, float, np.number))]
                    raise ValueError(f"无法将预测特征列 '{f_col}' 转为浮点数。错误值示例: {offending_values[:5]}. 原错误: {e_astype}")

        df_for_scaling_transformed = self.feature_scaler.transform(df_for_scaling) 
        df_scaled_features = pd.DataFrame(df_for_scaling_transformed, columns=expected_features_from_config, index=df_for_scaling.index)
        return df_scaled_features

    def predict(self, input_data_path_or_df):
        if isinstance(input_data_path_or_df, str):
            try: df_raw = pd.read_csv(input_data_path_or_df)
            except Exception: df_raw = pd.read_excel(input_data_path_or_df)
        elif isinstance(input_data_path_or_df, pd.DataFrame): df_raw = input_data_path_or_df.copy()
        else: raise ValueError("输入数据必须是文件路径(CSV/Excel)或Pandas DataFrame。")
        last_known_timestamp = None
        if isinstance(df_raw.index, pd.DatetimeIndex) and not df_raw.empty: last_known_timestamp = df_raw.index[-1]
        elif not df_raw.empty:
            time_cols = ['Time', 'timestamp', 'Datetime']; 
            for tc in time_cols:
                if tc in df_raw.columns: last_known_timestamp = pd.to_datetime(df_raw[tc].iloc[-1]); break
            if last_known_timestamp is None and 'date' in df_raw.columns and 'hour' in df_raw.columns:
                # 修正：确保 hour 列的值转换为字符串，然后再 zfill
                date_val = str(df_raw['date'].iloc[-1])
                hour_val = str(int(df_raw['hour'].iloc[-1])).zfill(2) # 假设hour是整数或可转为整数
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
        last_input_sequence_df = df_processed_features[model_input_feature_names].iloc[-self.model_config['look_back']:]
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

if __name__ == "__main__":
    print(f"使用设备: {DEVICE}"); os.makedirs(MODEL_ARTIFACTS_DIR, exist_ok=True) 
    action = input("您想做什么？ (1: 训练新模型, 2: 使用现有模型进行预测): ").strip()
    if action == '1':
        custom_artifacts_dir_train = input(f"请输入模型文件保存目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        os.makedirs(custom_artifacts_dir_train, exist_ok=True) 
        train_data_path = input(f"请输入训练数据文件路径 (默认为 '{DEFAULT_FILE_PATH}'): ").strip() or DEFAULT_FILE_PATH
        if not os.path.exists(train_data_path): print(f"错误: 训练数据文件 '{train_data_path}' 未找到。")
        else:
            train_config = {'file_path': train_data_path, 'look_back': DEFAULT_LOOK_BACK, 'horizon': DEFAULT_HORIZON, 
                            'target_col_names': DEFAULT_TARGET_COL_NAMES, 'primary_target_col_name': DEFAULT_PRIMARY_TARGET_COL_NAME,
                            'batch_size': DEFAULT_BATCH_SIZE, 'model_artifacts_dir': custom_artifacts_dir_train, 
                            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS, 'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
                            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS, 'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
                            'min_delta': DEFAULT_MIN_DELTA}
            trainer = ModelTrainer(train_config)
            try: trainer.run_training_pipeline()
            except Exception as e: print(f"训练过程中发生错误: {e}"); import traceback; traceback.print_exc()
    elif action == '2':
        custom_artifacts_dir_predict = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        try:
            predictor = ModelPredictor(custom_artifacts_dir_predict) 
            input_data_source = input("请输入用于预测的数据文件路径 (CSV 或 Excel): ").strip()
            if not os.path.exists(input_data_source): print(f"错误: 预测数据文件 '{input_data_source}' 未找到。")
            else:
                predicted_values, last_timestamp = predictor.predict(input_data_source)
                if predicted_values is not None and last_timestamp is not None:
                    print(f"\n预测的 {predictor.model_config['horizon']} 小时指标值 (原始尺度):")
                    for h in range(min(5, predictor.model_config['horizon'])):
                        hour_str = f"  小时 {h+1}: "; 
                        for t_idx, t_name in enumerate(predictor.model_config['target_col_names']): 
                            if t_name == 'CO':
                                hour_str += f"{t_name}={predicted_values[h, t_idx]:.2f} "
                            else:
                                hour_str += f"{t_name}={np.round(predicted_values[h, t_idx]).astype(int)} "
                        print(hour_str)
                    save_pred = input("是否将预测结果保存到CSV文件? (y/n): ").strip().lower() 
                    if save_pred.startswith('y'): 
                        pred_save_path = os.path.join(custom_artifacts_dir_predict, "predictions_output_adv.csv") 
                        future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=predictor.model_config['horizon'], freq='H')
                        output_data = {'date': future_timestamps.strftime('%Y%m%d'), 'hour': future_timestamps.hour}
                        
                        for t_idx, t_name in enumerate(predictor.model_config['target_col_names']): 
                            if t_name == 'CO':
                                output_data[t_name] = np.round(predicted_values[:, t_idx], 2)
                            else:
                                output_data[t_name] = np.round(predicted_values[:, t_idx]).astype(int)
                                
                        requested_output_columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
                        for col in requested_output_columns:
                            if col not in output_data: output_data[col] = [np.nan] * predictor.model_config['horizon']
                        output_df = pd.DataFrame(output_data)[requested_output_columns] 
                        
                        # 保存CSV时，Pandas会根据列的dtype自动格式化。我们已经处理了数据类型。
                        output_df.to_csv(pred_save_path, index=False, encoding='utf-8-sig') 
                        print(f"预测结果已保存到: {pred_save_path}")
                        predicted_cols_in_output = [col for col in predictor.model_config['target_col_names'] if col in requested_output_columns]
                        nan_cols_in_output = [col for col in requested_output_columns if col not in ['date', 'hour'] and col not in predicted_cols_in_output]
                        print(f"\n注意: CSV 文件中实际预测的列为: {', '.join(predicted_cols_in_output)}。")
                        if nan_cols_in_output: print(f"以下列填充为 NaN: {', '.join(nan_cols_in_output)}。")
                    else: print("预测结果未保存。")
                elif predicted_values is None: print("未能生成预测。")
                elif last_timestamp is None and predicted_values is not None : 
                     print("预测已生成，但无法从输入数据中确定最后一个时间戳。原始预测值 (horizon, num_targets):"); print(np.round(predicted_values,2))
        except FileNotFoundError as e: print(f"错误: {e}。")
        except ValueError as e: print(f"预测过程中发生值错误: {e}")
        except Exception as e: print(f"预测过程中发生未知错误: {e}"); import traceback; traceback.print_exc()
    else: print("无效的选择。请输入 '1' 或 '2'。")
    print("\nAQI预测流程结束。")

