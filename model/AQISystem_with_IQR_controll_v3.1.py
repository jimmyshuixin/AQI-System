# ==============================================================================
# 核心功能:
# 该脚本构建了一个综合系统，利用Transformer神经网络模型对多种空气质量指标（包括AQI和各项污染物浓度）
# 进行多目标预测，并集成了异常数据检测功能。
#
# 系统组件与特性:
# 1. 数据ETL与高级预处理:
#    - 精确的时间戳解析与DataFrame索引构建。
#    - 针对数值转换的鲁棒性设计，以及全面的缺失值（NaN）填充机制。
#    - 训练阶段采用基于IQR（四分位距）的异常值识别与插值平滑技术（可通过全局开关控制）。
#    - 高级特征工程:
#        - 依据国家环境空气质量标准（GB 3095-2012）动态计算AQI，作为关键输入特征 ("AQI_calculated")。
#        - 生成时间周期性特征（例如，小时、星期、月份的正余弦编码），捕捉时间模式。
#        - 为目标变量及关键周期性特征（如 *_24h, *_8h 平均浓度列）构建滞后特征，引入历史依赖性。
#
# 2. 模型训练 (通过 AQISystem -> ModelTrainer 模块):
#    - 采用基于PyTorch框架实现的Transformer深度学习模型架构。
#    - 支持对多个空气质量目标（如AQI, PM2.5, CO等）进行同步预测。
#    - 整合Optuna库进行高效的超参数自动化搜索与优化。
#    - 训练过程中应用学习率动态调整策略（ReduceLROnPlateau）与早停（Early Stopping）机制，防止过拟合，提升泛化能力。
#
# 3. 模型预测 (通过 AQISystem -> ModelPredictor 逻辑集成):
#    - 便捷加载预先训练完成的模型权重及相关的预处理组件（如数据缩放器、模型配置）。
#    - 对新的外部输入数据执行空气质量预测。
#
# 4. 异常数据检测 (通过 AQISystem 模块):
#    - 利用训练好的模型对输入数据序列中的潜在异常点进行识别。
#    - 异常判断基于模型预测输出与实际观测值之间的残差，并结合统计学阈值。
#    - 提供异常点的可视化展示，辅助数据质量分析。
#
# 5. 命令行用户交互界面:
#    - 脚本启动时，通过命令行提示引导用户选择执行模型训练、批量预测或异常数据检测等核心功能。
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
# 尝试配置matplotlib以支持中文标签显示（依赖系统中SimHei字体的可用性）。
# axes.unicode_minus = False 确保负号可以正确显示。
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
except Exception as e:
    print(f"警告: Matplotlib中文字体设置失败: {e}。图表中的中文标签可能无法正常显示。")

# --- 全局核心参数定义 ---
DEFAULT_FILE_PATH = 'data_process\output\南京_AQI_Data.xlsx' # 默认训练数据文件的相对或绝对路径
DEFAULT_LOOK_BACK = 72 # 模型回溯历史数据的时间窗口长度（单位：小时）别太大，当心电脑爆炸
DEFAULT_HORIZON = 72   # 模型向前预测的时间范围（单位：小时）
# 定义模型需要预测的空气质量目标污染物列表
DEFAULT_TARGET_COL_NAMES = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
# 数据集中所有可能涉及的列名，主要用于预处理流程中的列识别与筛选
DEFAULT_ALL_AVAILABLE_COL_NAMES = [ 
    'date', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
    'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 
    'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'
]
# 在Optuna超参数优化及早停机制中，作为主要性能评估依据的目标污染物（通常选择综合性的AQI）
DEFAULT_PRIMARY_TARGET_COL_NAME = 'AQI' 
DEFAULT_BATCH_SIZE = 32 # 神经网络训练和评估时每个批次处理的数据样本量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动检测并选择可用的计算设备（优先GPU）

# --- 模型及相关文件保存路径与命名规范 ---
MODEL_ARTIFACTS_DIR = "model\output" # 存储所有模型相关文件（权重、配置、缩放器等）的目录名
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model_adv.pth" # 保存性能最佳的模型权重状态的文件名
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler_adv.pkl" # 保存特征数据缩放器的文件名
TARGET_SCALERS_SAVE_NAME = "aqi_target_scalers_adv.pkl" # 保存目标数据缩放器（通常为每个目标一个）字典的文件名
MODEL_CONFIG_SAVE_NAME = "model_config_adv.json" # 保存模型架构参数、列名配置等重要元数据的文件名

# --- 训练过程特定参数 (参数设置旨在追求较高的预测准确度) ---
DEFAULT_FULL_TRAIN_EPOCHS = 200  # 最终选定超参数后，模型进行完整训练的总轮数
DEFAULT_N_OPTUNA_TRIALS = 150    # Optuna进行超参数搜索时尝试的试验次数
DEFAULT_OPTUNA_EPOCHS = 30     # Optuna单次超参数试验中，模型训练的轮数（用于快速评估该组超参数的潜力）
DEFAULT_EARLY_STOPPING_PATIENCE = 20 # 早停机制的耐心轮数：验证集性能连续多少轮无显著提升则提前终止训练
DEFAULT_MIN_DELTA = 0.00001 # 判断验证集性能是否有“显著”提升的最小变化阈值（用于早停和模型保存）
DEFAULT_ANOMALY_THRESHOLD_FACTOR = 3.0 # 异常检测中，判断数据点是否异常的阈值因子（通常为预测误差均值加上N倍标准差）
DEFAULT_ENABLE_IQR_OUTLIER_DETECTION = False # 新增：IQR异常值检测功能的全局开关

# --- AQI (空气质量指数) 计算相关的国家标准常量 ---
# IAQI (Individual Air Quality Index，个体空气质量指数) 的等级划分标准值
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
# 各类污染物浓度对应的IAQI计算分段点 (单位: µg/m³，CO为mg/m³)
# 这些分段点依据中国《环境空气质量标准》(GB 3095-2012)，并对应特定的污染物浓度平均周期。
POLLUTANT_BREAKPOINTS = {
    'SO2_24h':   [0, 50,  150,  475,  800,  1600, 2100, 2620], # 二氧化硫 24小时平均浓度限值
    'NO2_24h':   [0, 40,  80,   180,  280,  565,  750,  940],  # 二氧化氮 24小时平均浓度限值
    'PM10_24h':  [0, 50,  150,  250,  350,  420,  500,  600],  # PM10 24小时平均浓度限值
    'CO_24h':    [0, 2,   4,    14,   24,   36,   48,   60],   # 一氧化碳 24小时平均浓度限值 (mg/m³)
    'O3_8h_24h': [0, 100, 160,  215,  265,  800, 1000, 1200],  # 臭氧 日最大8小时滑动平均浓度限值
    'O3_1h':     [0, 160, 200,  300,  400,  800, 1000, 1200],  # 臭氧 1小时平均浓度限值 (作为备用或特定场景下的计算依据)
    'PM2.5_24h': [0, 35,  75,   115,  150,  250,  350,  500]   # PM2.5 24小时平均浓度限值
}
# 为原始污染物（通常是小时浓度值）定义一个近似的IAQI计算分段点表。
# 此表主要用于在缺乏特定周期平均值（如24小时均值）数据时，提供一个基于小时值的回退计算方案。
# 注意：这些小时值分段点可能与官方标准中的小时限值不完全一致，仅作为近似处理或应急方案。
POLLUTANT_BREAKPOINTS_HOURLY_APPROX = {
    'SO2':   [0, 150, 500,  650,  800], # 二氧化硫 1小时平均浓度 (µg/m³) - 示例值，实际应用时应查阅官方小时标准
    'NO2':   [0, 100, 200,  700, 1200], # 二氧化氮 1小时平均浓度 (µg/m³) - 示例值
    'PM10':  [0, 50, 150, 250, 350, 420], # PM10 1小时浓度 (官方标准中可能无直接小时IAQI计算方法，常参考24h均值标准)
    'CO':    [0, 5,  10,   35,   60],    # 一氧化碳 1小时平均浓度 (mg/m³) - 示例值
    'O3':    POLLUTANT_BREAKPOINTS['O3_1h'], # 臭氧 1小时平均浓度 (µg/m³)，直接引用O3_1h标准
    'PM2.5': [0, 35, 75, 115, 150, 250] # PM2.5 1小时浓度 (官方标准中可能无直接小时IAQI计算方法，常参考24h均值标准)
}


# --- 辅助工具函数 ---
def set_seed(seed_value=42):
    """
    设定全局随机种子，确保实验（如模型训练、数据划分等）的可复现性。
    Args:
        seed_value (int): 随机种子值。
    """
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # 以下两项设置有助于确保CUDA操作在多次运行时产生相同的结果，但可能会牺牲一些性能。
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
    os.environ['PYTHONHASHSEED'] = str(seed_value) # 设置Python内置哈希函数的种子

def calculate_iaqi(Cp, pollutant_key):
    """
    根据污染物浓度计算其对应的个体空气质量指数 (IAQI)。
    IAQI的计算遵循中国环境保护标准中规定的分段线性插值方法。

    Args:
        Cp (float): 待计算的污染物浓度值。
        pollutant_key (str): 污染物类型及其浓度周期的标识符 (例如 'PM2.5_24h', 'O3_1h')。
                             该键名用于在POLLUTANT_BREAKPOINTS或POLLUTANT_BREAKPOINTS_HOURLY_APPROX
                             查找对应的浓度限值分段点。
    Returns:
        float: 计算得到的IAQI值。如果输入浓度无效或未找到对应的污染物标准，则返回np.nan。
    """
    if pd.isna(Cp) or Cp < 0: # 处理无效的浓度输入
        return np.nan 

    # 优先使用官方定义的特定周期浓度限值表
    bp_table_to_use = POLLUTANT_BREAKPOINTS 
    if pollutant_key not in bp_table_to_use:
        # 如果特定周期标准不存在，则尝试使用小时浓度近似标准表作为回退
        bp_table_to_use = POLLUTANT_BREAKPOINTS_HOURLY_APPROX 
        if pollutant_key not in bp_table_to_use:
            # 若两种标准表均未找到对应污染物，则无法计算IAQI
            # print(f"警告: 未找到污染物 '{pollutant_key}' 的浓度限值表。无法计算其IAQI。") # 此日志可能过于频繁，视情况启用
            return np.nan
            
    bp = bp_table_to_use.get(pollutant_key) # 获取该污染物的浓度分段点列表

    # IAQI计算核心逻辑：分段线性插值
    if Cp > bp[-1]: # 如果浓度超过最高分段点，IAQI通常封顶为500（或根据标准进行外插，此处简化为500）
        return 500 
    for i in range(len(bp) - 1):
        # 判断浓度Cp落在哪个分段区间 [BP_Lo, BP_Hi)
        if bp[i] <= Cp < bp[i+1]: 
            IAQI_Lo, IAQI_Hi = IAQI_LEVELS[i], IAQI_LEVELS[i+1] # 获取对应区间的IAQI上下限
            BP_Lo, BP_Hi = bp[i], bp[i+1] # 获取对应区间的浓度上下限
            if BP_Hi == BP_Lo: return IAQI_Lo # 防止除以零错误（理论上标准分段点不会相同）
            # 应用线性插值公式计算IAQI
            return round(((IAQI_Hi - IAQI_Lo) / (BP_Hi - BP_Lo)) * (Cp - BP_Lo) + IAQI_Lo)
    if Cp == bp[0]: # 如果浓度恰好等于最低分段点
        return IAQI_LEVELS[0]
    return np.nan # 理论上，如果标准定义完整，不应执行到此处

def calculate_aqi_from_pollutants(df):
    """
    根据输入的DataFrame中包含的多种污染物浓度数据，计算综合空气质量指数 (AQI)，
    并识别造成当前AQI水平的首要污染物。
    计算过程优先选用数据中提供的特定周期平均浓度列（如 PM2.5_24h）。

    Args:
        df (pd.DataFrame): 包含污染物浓度数据的DataFrame。期望的列名如 'PM2.5', 'PM2.5_24h' 等。
    Returns:
        pd.DataFrame: 在输入DataFrame基础上增加了 'AQI_calculated' (计算得到的AQI) 
                      和 'Primary_Pollutant_calculated' (计算得到的首要污染物) 两列。
    """
    iaqi_df = pd.DataFrame(index=df.index) # 创建临时DataFrame存储各污染物的IAQI值

    # 定义用于计算综合AQI的污染物及其在输入DataFrame中的首选和备选数据列名。
    # 格式: {标准IAQI计算键名: [DataFrame中的首选列, DataFrame中的次选列, ...]}
    # 例如，对于'PM2.5_24h'的IAQI计算，优先使用'PM2.5_24h'列，若不存在或全为NaN，则尝试使用'PM2.5'列。
    pollutants_for_calc = {
        'SO2_24h':   ['SO2_24h', 'SO2'], 
        'NO2_24h':   ['NO2_24h', 'NO2'],
        'PM10_24h':  ['PM10_24h', 'PM10'], 
        'CO_24h':    ['CO_24h', 'CO'],
        'O3_8h_24h': ['O3_8h_24h', 'O3_8h', 'O3'], # 臭氧的优先级：日最大8h滑动平均 -> 8h滑动平均 -> 1h平均
        'PM2.5_24h': ['PM2.5_24h', 'PM2.5']
    }

    # 针对每种标准污染物，选择合适的数据列计算其IAQI
    for bp_key, df_col_options in pollutants_for_calc.items():
        selected_col_for_iaqi = None
        # 遍历列选项，找到第一个在DataFrame中存在且包含有效数据的数据列
        for df_col in df_col_options: 
            if df_col in df.columns and not df[df_col].isnull().all():
                selected_col_for_iaqi = df_col
                break 
        
        if selected_col_for_iaqi:
            # 使用bp_key（如'PM2.5_24h'）作为污染物类型标识查找浓度限值表，
            # 使用selected_col_for_iaqi（如'PM2.5_24h'或'PM2.5'）从DataFrame中获取实际浓度数据。
            iaqi_df[bp_key] = df[selected_col_for_iaqi].apply(lambda x: calculate_iaqi(x, bp_key if bp_key in POLLUTANT_BREAKPOINTS else selected_col_for_iaqi))
        else:
            # 如果所有选项列都不可用或全为NaN，则该污染物的IAQI记为NaN
            iaqi_df[bp_key] = np.nan 
            
    # 根据标准，AQI取所有参与计算的污染物IAQI中的最大值
    df['AQI_calculated'] = iaqi_df.max(axis=1, skipna=True)
    
    # 确定首要污染物：当AQI > 50时，首要污染物是IAQI值等于最终AQI值的那些污染物。
    def get_primary_pollutants(row):
        # AQI小于等于50时，空气质量为优或良，无首要污染物。
        if pd.isna(row['AQI_calculated']) or row['AQI_calculated'] <= 50: return '无' 
        # 找出所有IAQI值（四舍五入后）等于最终AQI值（四舍五入后）的污染物
        primary = [pollutant_bp_key for pollutant_bp_key in iaqi_df.columns 
                   if pd.notna(row[pollutant_bp_key]) and round(row[pollutant_bp_key]) == round(row['AQI_calculated'])]
        return ', '.join(primary) if primary else '无' # 如果有多个，则用逗号分隔

    # 为确定首要污染物，需要将计算出的AQI值临时加入到包含各IAQI的DataFrame中，以便apply函数能同时访问它们。
    temp_iaqi_df_for_primary = iaqi_df.copy()
    temp_iaqi_df_for_primary['AQI_calculated'] = df['AQI_calculated'] # 将刚计算的AQI加入
    df['Primary_Pollutant_calculated'] = temp_iaqi_df_for_primary.apply(get_primary_pollutants, axis=1)
    
    return df

def create_sequences(data_df, look_back, horizon, target_col_names, feature_cols, is_predict=False):
    """
    从预处理后的时间序列DataFrame中创建适用于序列模型（如Transformer, LSTM）的输入序列 (X) 和输出序列 (y)。

    Args:
        data_df (pd.DataFrame): 包含所有特征和目标列的时间序列数据，通常已经过缩放。
        look_back (int): 输入序列的长度，即模型回溯参考的历史时间步数量。
        horizon (int): 输出序列的长度，即模型需要预测的未来时间步数量。
        target_col_names (list): 需要预测的目标列的名称列表。
        feature_cols (list): 用作模型输入的特征列的名称列表。
        is_predict (bool, optional): 是否为预测模式。
                                     若为True，则只生成输入序列X，不生成y（因为未来值未知）。
                                     默认为False（训练或评估模式）。
    Returns:
        tuple: 包含两个NumPy数组 (X_arr, y_arr)。
               X_arr 的形状为 (num_sequences, look_back, num_features)。
               y_arr 的形状为 (num_sequences, horizon, num_targets)。
               如果 is_predict=True，则返回 (X_arr, None)。
               如果数据不足以生成任何有效序列，则返回空的NumPy数组。
    Raises:
        ValueError: 如果指定的特征列或目标列在data_df中不存在。
    """
    X_list, y_list = [], [] # 用于存储生成的序列片段
    # 校验特征列和目标列是否存在于输入DataFrame中
    missing_feature_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_feature_cols: raise ValueError(f"数据DataFrame中缺少必要的特征列: {missing_feature_cols}.")
    if not is_predict: # 仅在非预测模式下校验目标列
        missing_target_cols = [col for col in target_col_names if col not in data_df.columns]
        if missing_target_cols: raise ValueError(f"数据DataFrame中缺少必要的目标列: {missing_target_cols}.")

    # 将选择的特征和目标列转换为NumPy数组以提高切片效率
    data_features_np = data_df[feature_cols].values
    if not is_predict: data_targets_np = data_df[target_col_names].values
    
    num_samples = len(data_features_np) # 总样本数（时间点数）
    # 计算可以从数据中生成的完整序列的数量
    if is_predict: # 预测模式下，只需要look_back长度的X，不需要为y的horizon留出空间
        num_possible_sequences = num_samples - look_back + 1
    else: # 训练/评估模式下，需要为y的horizon在look_back之后留出足够的空间
        num_possible_sequences = num_samples - look_back - horizon + 1

    if num_possible_sequences <= 0: # 如果数据太短，无法生成任何完整序列
        # 返回形状正确的空数组，以便后续代码可以安全地检查 .size 或 .shape[0]
        num_features = len(feature_cols)
        num_targets = len(target_col_names) if not is_predict else 0
        empty_x_shape = (0, look_back, num_features)
        empty_y_shape = (0, horizon, num_targets) if not is_predict else (0,) # y可以简单处理
        
        return np.empty(empty_x_shape), (np.empty(empty_y_shape) if not is_predict else None)


    # 遍历数据，按滑动窗口方式生成X和y序列
    for i in range(num_possible_sequences):
        X_list.append(data_features_np[i : i + look_back]) # 输入序列X
        if not is_predict:
            y_list.append(data_targets_np[i + look_back : i + look_back + horizon, :]) # 对应的输出序列y
            
    X_arr = np.array(X_list) if X_list else np.empty((0, look_back, len(feature_cols)))
    
    if is_predict: 
        return X_arr, None # 预测模式只返回X
        
    y_arr = np.array(y_list) if y_list else np.empty((0, horizon, len(target_col_names)))
    
    # 在非预测模式下，如果X已创建但y为空（通常因数据末尾不足以形成完整horizon的y），
    # 这表示这些X序列没有对应的y，应视作无效，返回空数组。
    if y_arr.size == 0 and X_arr.size > 0 and not is_predict : 
        print("警告: 已创建输入(X)序列，但由于数据末端长度不足，未能生成对应的输出(y)序列。")
        return np.empty((0, look_back, len(feature_cols))), np.empty((0, horizon, len(target_col_names)))
        
    return X_arr, y_arr


def plot_training_loss(train_losses, val_losses, save_path, title_prefix=""):
    """
    绘制并保存模型训练过程中的训练损失和验证损失曲线。
    Args:
        train_losses (list): 包含每个epoch训练损失的列表。
        val_losses (list): 包含每个epoch验证损失的列表。
        save_path (str): 损失曲线图的保存路径。
        title_prefix (str, optional): 图表标题的前缀。默认为空。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失 (Train Loss)')
    plt.plot(val_losses, label='验证损失 (Validation Loss)')
    plt.title(f'{title_prefix}模型训练过程中的损失变化')
    plt.xlabel('训练轮数 (Epoch)')
    plt.ylabel('损失函数值 (Loss - MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close() # 关闭图像，释放资源

def plot_predictions_vs_actual(actual, predicted, target_name, save_path_prefix, title_suffix="实际值 vs. 预测值"):
    """
    为单个目标变量绘制并保存其实际值与模型预测值的对比图。
    Args:
        actual (np.ndarray): 目标变量的实际值数组 (通常为1D或可展平)。
        predicted (np.ndarray): 目标变量的模型预测值数组 (形状应与actual一致)。
        target_name (str): 目标变量的名称 (用于图表标题和标签)。
        save_path_prefix (str): 保存图表文件的前缀 (会自动附加目标名称和后缀)。
        title_suffix (str, optional): 图表标题的后缀。默认为 "实际值 vs. 预测值"。
    """
    plt.figure(figsize=(15, 7))
    actual_flat = actual.flatten() # 确保数据是一维的
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
    """
    为单个目标变量绘制时间序列图，并在图上标记识别出的异常点。
    Args:
        timestamps_or_indices (array-like): 与actual_values对应的时间戳或索引数组。
        actual_values (np.ndarray): 目标变量的实际值数组 (通常为1D或可展平)。
        anomaly_indices (array-like): 标记为异常点在actual_values中的索引列表。
        target_name (str): 目标变量的名称。
        save_path_prefix (str): 保存图表文件的前缀。
        title_suffix (str, optional): 图表标题的后缀。默认为 "异常点"。
    """
    plt.figure(figsize=(15, 7))
    actual_flat = actual_values.flatten() # 确保数据是一维的

    # 绘制原始时间序列
    plt.plot(timestamps_or_indices, actual_flat, label=f'实际值 ({target_name})', alpha=0.7)
    
    # 筛选出在 actual_flat 长度范围内的有效异常点索引
    valid_anomaly_indices = np.array([idx for idx in anomaly_indices if 0 <= idx < len(actual_flat)], dtype=int)

    if len(valid_anomaly_indices) > 0:
        # 根据 timestamps_or_indices 的类型准备绘图的X轴数据
        if isinstance(timestamps_or_indices, (pd.DatetimeIndex, pd.RangeIndex)):
            # 如果是Pandas的索引类型，直接使用
            plot_x_values = np.array(timestamps_or_indices)
        else: 
            # 否则，确保是NumPy数组
            plot_x_values = np.asarray(timestamps_or_indices)

        # 获取异常点对应的X（时间/索引）和Y（实际值）
        anomaly_x = plot_x_values[valid_anomaly_indices]
        anomaly_y = actual_flat[valid_anomaly_indices]
        # 在图上用红色散点标记异常点
        plt.scatter(anomaly_x, anomaly_y, color='red', label='检测到的异常点 (Anomaly)', marker='o', s=50, zorder=5)

    plt.title(f'{target_name} - {title_suffix}')
    plt.xlabel('时间 / 序列索引') 
    plt.ylabel(f'{target_name} 浓度/指数值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
    save_full_path = f"{save_path_prefix}_anomalies.png" # 构建完整保存路径
    plt.savefig(save_full_path)
    plt.close()


def detect_anomalies_iqr_and_impute(df, column_names, factor=1.5, interpolation_method='time'):
    """
    （主要用于训练数据预处理阶段）
    使用IQR（四分位距）方法检测指定数值列中的统计学异常点，并将这些异常点替换为NaN，
    随后尝试通过插值方法填充这些NaN以及原有的NaN。
    对于插值后仍存在的NaN（通常发生在序列的开始或结束位置），则使用该列的中位数进行填充。

    Args:
        df (pd.DataFrame): 输入的DataFrame。
        column_names (list): 需要进行异常检测和插值填充的列名列表。
        factor (float, optional): IQR的倍数，用于定义异常值的边界。默认为1.5。
                                  （即：Q1 - factor*IQR, Q3 + factor*IQR 之外的值被视为异常）
        interpolation_method (str, optional): 传递给 pd.DataFrame.interpolate() 的插值方法。
                                             如果DataFrame索引是DatetimeIndex，默认为'time'（基于时间权重的插值）；
                                             否则，通常回退到'linear'。
    Returns:
        pd.DataFrame: 处理了指定列中异常值和缺失值的DataFrame副本。
    """
    df_cleaned = df.copy() # 创建副本以避免修改原始DataFrame
    print("开始对指定列进行基于IQR的异常值检测和插值填充...")
    for col_name in column_names:
        if col_name in df_cleaned.columns:
            # 确保操作的列是数值类型，否则IQR和插值无法有效执行
            if not pd.api.types.is_numeric_dtype(df_cleaned[col_name]):
                print(f"警告: 列 '{col_name}' 非数值类型，跳过其异常值处理流程。")
                continue

            original_nan_count = df_cleaned[col_name].isna().sum() # 记录原始NaN数量（仅供参考）
            # 计算第一四分位数(Q1)和第三四分位数(Q3)
            Q1 = df_cleaned[col_name].quantile(0.25)
            Q3 = df_cleaned[col_name].quantile(0.75)
            IQR = Q3 - Q1 # 计算四分位距

            # 仅当IQR有效（非NaN且大于一个极小值，避免IQR为0或接近0时误判）时，才进行基于IQR的异常检测
            if pd.notna(IQR) and IQR > 1e-6: # 1e-6是一个小的阈值，防止IQR过小导致边界过于敏感
                lower_bound = Q1 - factor * IQR # 定义异常值下界
                upper_bound = Q3 + factor * IQR # 定义异常值上界
                # 识别超出边界的异常值
                outlier_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
                num_outliers = outlier_mask.sum()
                if num_outliers > 0: 
                    print(f"列 '{col_name}': 检测到 {num_outliers} 个基于IQR的统计学异常值。将其标记为NaN以便后续插值。")
                    df_cleaned.loc[outlier_mask, col_name] = np.nan # 将识别出的异常值替换为NaN
            else:
                print(f"列 '{col_name}': IQR值为0或无效 ({IQR})，跳过基于IQR的异常值标记步骤。")

            # 对列中的NaN值（包括原始的和因异常值标记产生的）进行插值
            if isinstance(df_cleaned.index, pd.DatetimeIndex): # 如果索引是时间类型，优先使用时间插值
                try: 
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method=interpolation_method, limit_direction='both')
                except Exception as e: 
                    print(f"列 '{col_name}' 时间插值失败: {e}。尝试使用线性插值作为备选。")
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            else: # 如果不是时间索引，使用线性插值
                df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            
            # 填充插值后可能仍然存在的NaN值（通常是序列开头或结尾无法插值的点）
            if df_cleaned[col_name].isna().sum() > 0 : 
                median_val = df_cleaned[col_name].median() # 使用当前列（可能已部分插值）的中位数
                # 如果中位数本身也是NaN（例如，列中所有值都是NaN或异常值），则用0填充
                fill_value = median_val if pd.notna(median_val) else 0 
                df_cleaned[col_name] = df_cleaned[col_name].fillna(fill_value)
                print(f"列 '{col_name}': 剩余的NaN值已使用中位数({median_val:.2f})或0 (若中位数无效，则为{fill_value:.2f})进行填充。")
        else: 
            print(f"警告: 列 '{col_name}' 在DataFrame中未找到，跳过其异常值处理。")
    print("基于IQR的异常值检测和插值填充流程完成。")
    return df_cleaned

class TimeSeriesDataset(TensorDataset):
    """
    自定义PyTorch数据集类，用于封装时间序列的输入(X)和目标(y)张量。
    继承自torch.utils.data.TensorDataset，简化了数据加载器的使用。
    """
    def __init__(self, X, y): 
        # 将NumPy数组转换为PyTorch张量，并确保数据类型为float
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        # 调用父类的构造函数
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor)

class AQITransformer(nn.Module):
    """
    基于Transformer Encoder架构的多目标时间序列预测神经网络模型。

    Args:
        num_features (int): 输入序列中每个时间步的特征数量。
        d_model (int): Transformer模型内部的特征维度 (也称为embedding维度)。
        nhead (int): 多头注意力机制（Multi-Head Attention）中的头数。d_model必须能被nhead整除。
        num_encoder_layers (int): Transformer编码器中堆叠的编码器层（EncoderLayer）的数量。
        dim_feedforward (int): 编码器层中前馈神经网络（Feedforward Network）的内部维度。
        dropout (float): 在模型多个位置应用的Dropout比率，用于正则化，防止过拟合。
        horizon (int): 模型需要预测的未来时间步数量。
        num_target_features (int): 需要同时预测的目标变量的数量。
        norm_first (bool, optional): 是否在Transformer编码器层中先应用层归一化（Layer Normalization），
                                     再进行其他操作（如自注意力和前馈网络）。
                                     设置为True通常能带来更稳定的训练。默认为True。
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, num_target_features, norm_first=True):
        super(AQITransformer, self).__init__()
        self.d_model = d_model # Transformer内部维度
        self.horizon = horizon # 预测范围
        self.num_target_features = num_target_features # 目标数量

        # 输入嵌入层: 将原始特征维度的输入映射到Transformer的内部维度d_model
        self.input_embedding = nn.Linear(num_features, d_model) 
        
        # 定义单个Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu', # 使用GELU激活函数，一种平滑的ReLU变体
            batch_first=True,  # 输入和输出张量的形状为 (batch, seq_len, features)
            norm_first=norm_first # 推荐设置为True以提高训练稳定性
        )
        # 将多个编码器层堆叠起来形成完整的Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # 输出层: 将Transformer编码器最终的输出（通常是最后一个时间步的表示）
        # 映射到预测结果的维度 (horizon * num_target_features)，即所有目标在整个预测范围内的值。
        self.output_layer = nn.Linear(d_model, horizon * num_target_features) 
    
    def forward(self, src):
        """
        模型的前向传播过程。
        Args:
            src (torch.Tensor): 输入序列张量，形状为 (batch_size, seq_len, num_features)。
        Returns:
            torch.Tensor: 模型的预测输出张量，形状为 (batch_size, horizon, num_target_features)。
        """
        # 1. 输入嵌入与缩放
        # src 形状: (batch_size, seq_len, num_features)
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model) # 嵌入并按d_model的平方根缩放，常见做法
        # src_embedded 形状: (batch_size, seq_len, d_model)
        
        # 2. 简化的正弦位置编码 (Sinusoidal Positional Encoding)
        # 为序列中的每个位置添加独特的位置信息，使模型能够区分不同位置的元素。
        seq_len = src_embedded.size(1) # 获取输入序列的实际长度
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device) # 初始化位置编码张量
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device) # 位置索引 (0, 1, ..., seq_len-1)
        # 计算用于正弦和余弦函数的分母项 (div_term)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度使用sin
        # 奇数维度使用cos，注意处理d_model为奇数时最后一个维度可能无法完整配对的情况
        if self.d_model % 2 != 0: 
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)] 
        else: 
            pe[:, 1::2] = torch.cos(position * div_term)
        
        src_pos_encoded = src_embedded + pe.unsqueeze(0) # 将位置编码加到嵌入向量上 (广播机制)
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded) # 对加入位置编码后的嵌入向量应用Dropout
        # src_pos_encoded 形状: (batch_size, seq_len, d_model)
        
        # 3. Transformer编码器处理
        # 输入经过嵌入和位置编码后，送入Transformer编码器进行序列信息提取。
        encoder_output = self.transformer_encoder(src_pos_encoded) 
        # encoder_output 形状: (batch_size, seq_len, d_model)
        
        # 4. 从编码器输出中提取用于预测的表示
        # 通常使用编码器输出序列的最后一个时间步的隐藏状态作为整个输入序列的聚合表示。
        prediction_input = encoder_output[:, -1, :] 
        # prediction_input 形状: (batch_size, d_model)
        
        # 5. 输出层预测
        output_flat = self.output_layer(prediction_input) 
        # output_flat 形状: (batch_size, horizon * num_targets)，是扁平化的预测结果
        
        # 6. 重塑输出以匹配多目标、多步预测的期望格式
        # 将扁平化的输出重塑为 (batch_size, horizon, num_target_features)
        output = output_flat.view(output_flat.size(0), self.horizon, self.num_target_features)
        return output

class ModelTrainer:
    """
    该类封装了模型训练的完整流程，包括数据加载、预处理、超参数优化（使用Optuna）、
    模型的核心训练循环、性能评估以及最终模型和相关组件（如缩放器、配置）的保存。
    """
    def __init__(self, config):
        """
        初始化ModelTrainer实例。
        Args:
            config (dict): 包含所有训练相关配置参数的字典。
                           这些参数由AQISystem类在实例化ModelTrainer时提供。
        """
        self.config = config 
        # self.config['model_artifacts_dir'] 应由AQISystem确保已设置并创建
        # set_seed() 通常在AQISystem初始化时调用，以保证全局一致性，此处不再重复调用，
        # 除非有特定需求要在此处重设种子。

        self.all_feature_columns_for_sequence = [] # 用于存储最终确定用于创建序列的特征列名列表
        self.feature_scaler = None # 特征数据缩放器实例 (例如 StandardScaler)
        self.target_scalers = {}   # 目标数据缩放器字典 (键为目标列名，值为对应的缩放器实例)

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True):
        """
        核心的数据加载和预处理函数。负责从原始数据文件到模型可用数据的完整转换流程。
        主要步骤包括：
        1.  数据读取（CSV或Excel）。
        2.  时间戳解析与DataFrame索引设置。
        3.  强制数值类型转换，非数值转为NaN。
        4.  缺失值（NaN）填充：对特征列采用前向、后向再用0填充的策略；
            对目标列（训练时）则直接删除包含NaN的行，以保证训练数据的质量。
        5.  （仅训练时且全局开关开启时）基于IQR的异常值检测与插值修复：针对目标列进行。
        6.  特征工程：
            a.  计算AQI (AQI_calculated)。
            b.  创建周期性时间特征（如小时、星期的正余弦编码）。
            c.  为目标列和关键的周期性平均值列创建滞后特征。
        7.  确定最终用于模型输入的特征列列表。
        8.  （仅训练时或fit_scalers=True时）数据缩放：
            a.  为特征列拟合（fit）并应用（transform）一个StandardScaler。
            b.  为每个目标列单独拟合并应用一个StandardScaler。
            c.  保存拟合好的缩放器到文件。
        9.  （预测或评估时，fit_scalers=False）数据缩放：加载已保存的缩放器并应用（transform）。

        Args:
            file_path (str): 待加载数据文件的路径。
            fit_scalers (bool, optional): 是否需要拟合新的数据缩放器。
                                         在训练新模型时应为True；
                                         在对新数据进行预测或评估已训练模型时，应为False，此时会加载已保存的缩放器。
                                         默认为True。
        Returns:
            pd.DataFrame: 经过完整预处理（包括特征工程和数据缩放）后的DataFrame。
        """
        print(f"开始从文件加载并执行核心数据预处理流程: {file_path}...")
        # 1. 数据读取 (尝试CSV，失败则尝试Excel)
        try: 
            df = pd.read_csv(file_path)
        except Exception: # 捕获通用异常，因为具体错误可能多样
            try: 
                df = pd.read_excel(file_path)
            except Exception as e_excel: 
                print(f"错误: 读取CSV和Excel文件均失败。路径: '{file_path}'. 详细错误: {e_excel}")
                raise # 重新抛出异常，终止后续流程

        # 2. 时间戳处理和索引设置
        # 优先尝试基于 'date' 和 'hour' 列构建时间戳
        if 'date' in df.columns and 'hour' in df.columns: 
            try:
                # 将 'date' 和 'hour' 合并为标准时间戳格式
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(int).astype(str).str.zfill(2), format='%Y%m%d%H')
                df = df.set_index('timestamp').drop(columns=['date', 'hour'], errors='ignore') # 设置为索引并删除原始列
            except Exception as e_dt:
                print(f"警告: 从'date'和'hour'列创建时间戳索引失败: {e_dt}。尝试其他方法...")
                # 如果失败，后续逻辑会尝试其他时间列或第一列
        # 如果没有 'date'/'hour'，尝试 'Time' 列
        elif 'Time' in df.columns: 
             try:
                df['timestamp'] = pd.to_datetime(df['Time'])
                df = df.set_index('timestamp').drop(columns=['Time'], errors='ignore')
             except Exception as e_t:
                print(f"警告: 从'Time'列创建时间戳索引失败: {e_t}。尝试其他方法...")
        # 如果以上都不成功，尝试将DataFrame的第一列作为时间索引 (作为最后手段)
        else: 
            try: 
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception as e_first_col: 
                raise ValueError(f"错误: 无法自动从数据中解析并设置时间索引。请确保数据包含 ('date' 和 'hour') 列组合，或 'Time' 列，或第一列是可被pandas解析的时间格式。失败详情: {e_first_col}")

        # 3. 将所有数据列尝试转换为数值类型，无法转换的值设为NaN
        for col in df.columns: 
            df[col] = pd.to_numeric(df[col], errors='coerce') # 'coerce'会将无效解析值替换为NaN
        
        # 4. 处理非目标特征列中的NaN值 (采用 前向填充 -> 后向填充 -> 0填充 的策略)
        #    目标列中的NaN将在后续步骤中处理或导致行被删除（训练时）。
        feature_candidate_cols = [col for col in df.columns if col not in self.config['target_col_names']]
        for col in feature_candidate_cols: 
            if df[col].isnull().any(): # 只处理包含NaN的列
                df[col] = df[col].ffill().bfill().fillna(0) # 链式填充
        
        # 对于训练数据（fit_scalers=True），严格要求目标列不能有NaN。删除包含NaN的行。
        # 对于预测数据，目标列可能不存在或不用于此阶段，此dropna主要针对训练。
        if fit_scalers: # 通常在训练时为True
            original_len = len(df)
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
            if len(df) < original_len:
                print(f"信息: 由于目标列存在NaN值，已删除 {original_len - len(df)} 行数据。")
        if df.empty: 
            raise ValueError("错误: 数据在初步NaN处理（特别是目标列NaN行删除）后变为空。请检查原始数据质量。")

        # 5. 复杂异常值处理 (仅在训练并拟合新缩放器时，且全局开关开启时，对目标列进行IQR异常检测和插值)
        if fit_scalers and self.config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION): 
            print("信息: IQR异常值检测功能已启用。开始对目标列进行处理...")
            df = detect_anomalies_iqr_and_impute(df, self.config['target_col_names'])
            # 再次检查并删除可能因插值失败（如序列开头结尾）而仍存在的NaN行（针对目标列）
            original_len_after_iqr = len(df)
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names']) 
            if len(df) < original_len_after_iqr:
                 print(f"信息: 在IQR异常处理和插值后，由于目标列仍存在NaN，又删除了 {original_len_after_iqr - len(df)} 行。")
        elif fit_scalers: # 如果是训练模式但IQR开关关闭
            print("信息: IQR异常值检测功能已禁用。跳过对目标列的IQR处理步骤。")

        if df.empty: 
            raise ValueError("错误: 数据在IQR异常值处理流程（如果启用）或初步NaN处理后变为空。")

        # 6. AQI 计算 (特征工程核心步骤之一)
        # 使用更新后的AQI计算函数，它会尝试使用不同周期的污染物数据。
        # 注意：calculate_aqi_from_pollutants会修改传入的df副本（如果用df.copy()）或原df。
        # 此处确保传递副本，以防原始df在其他地方还需要。
        df = calculate_aqi_from_pollutants(df.copy()) 
        # 填充AQI计算后可能产生的NaN (例如，如果所有相关污染物浓度都缺失导致无法计算AQI)
        if 'AQI_calculated' in df.columns and df['AQI_calculated'].isnull().any():
            df['AQI_calculated'] = df['AQI_calculated'].fillna(0) # 用0填充无法计算的AQI


        # 7. 创建周期性时间特征 (优化为使用pd.concat一次性添加所有新周期特征列)
        new_cyclical_features = pd.DataFrame(index=df.index) # 创建空的DataFrame以收集新特征
        if isinstance(df.index, pd.DatetimeIndex): # 确保索引是时间类型才能提取hour, dayofweek等
            idx = df.index
            new_cyclical_features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['month_sin'] = np.sin(2 * np.pi * idx.month / 12.0) 
            new_cyclical_features['month_cos'] = np.cos(2 * np.pi * idx.month / 12.0)
            df = pd.concat([df, new_cyclical_features], axis=1) # 合并新特征到主DataFrame
        else:
            print("警告: DataFrame索引不是DatetimeIndex，无法创建周期性时间特征。")
        
        # 8. 创建滞后特征 (优化为使用pd.concat一次性添加所有新滞后特征列)
        # 滞后特征数量可以根据look_back窗口动态调整，例如取look_back的1/4，但至少为1。
        num_lags_to_create = max(1, self.config['look_back'] // 4) 
        lag_features_to_concat = [] # 存储待合并的滞后特征Series对象列表
        lag_cols_created_names = [] # 存储新创建的滞后特征列名，用于后续dropna操作

        # 为所有目标列以及所有以 _24h 或 _8h 结尾的列（通常代表周期平均浓度，是重要特征）创建滞后项。
        # 这样做可以为模型提供更丰富的历史信息。
        cols_for_lags = self.config['target_col_names'] + \
                        [col for col in df.columns if (col.endswith('_24h') or col.endswith('_8h')) and col not in self.config['target_col_names']]
        cols_for_lags = sorted(list(set(cols_for_lags))) # 去重并排序，确保列处理顺序的一致性

        for col_to_lag in cols_for_lags: 
            if col_to_lag in df.columns: # 确保原始列存在于DataFrame中
                for lag in range(1, num_lags_to_create + 1): # 创建指定数量的滞后项
                    lag_col_name = f"{col_to_lag}_lag_{lag}" # 构建滞后特征列名
                    lag_features_to_concat.append(df[col_to_lag].shift(lag).rename(lag_col_name))
                    lag_cols_created_names.append(lag_col_name)
        
        if lag_features_to_concat: # 如果成功创建了任何滞后特征
            df = pd.concat([df] + lag_features_to_concat, axis=1) # 一次性合并所有滞后特征
            # 删除因创建滞后特征而产生的NaN行（主要影响数据的前面部分）
            original_len_before_lag_dropna = len(df)
            df = df.dropna(subset=lag_cols_created_names, how='any') 
            if len(df) < original_len_before_lag_dropna:
                print(f"信息: 由于创建滞后特征引入NaN，已删除 {original_len_before_lag_dropna - len(df)} 行数据。")
        
        if df.empty: 
            raise ValueError("错误: 数据在创建滞后特征并处理相关NaN后变为空。")

        # 9. 定义最终的特征列列表 (self.all_feature_columns_for_sequence)
        # 这个列表包含了所有将用于创建模型输入序列X的特征列名。
        # 它不应包含原始的目标列名（以避免数据泄露），但可以包含目标列的滞后项、
        # 计算得到的AQI_calculated（如果它本身不是一个直接预测的目标）、周期性特征等。
        self.all_feature_columns_for_sequence = [col for col in df.columns if col not in self.config['target_col_names']]
        
        # 特殊处理 AQI_calculated 列：
        # 如果 'AQI_calculated' 存在于DataFrame中，并且它 *不* 是一个要直接预测的目标，
        # 那么它应该被视为一个输入特征。
        if 'AQI_calculated' in df.columns and 'AQI_calculated' not in self.config['target_col_names']:
            if 'AQI_calculated' not in self.all_feature_columns_for_sequence: # 确保不重复添加
                 self.all_feature_columns_for_sequence.append('AQI_calculated')
        # 一个潜在的配置问题：如果 'AQI_calculated' 同时被定义为目标和特征，这可能导致训练问题。
        elif 'AQI_calculated' in self.all_feature_columns_for_sequence and 'AQI_calculated' in self.config['target_col_names']:
             print("配置警告: 'AQI_calculated' 列同时被识别为模型输入特征和预测目标。这可能不符合预期，并可能导致数据泄露。请检查target_col_names配置。")
        
        self.all_feature_columns_for_sequence = sorted(list(set(self.all_feature_columns_for_sequence))) # 再次去重并排序，保证一致性

        # 10. 检查目标列在缩放前是否存在于DataFrame中，这是必要的。
        missing_targets_for_scaling = [tc for tc in self.config['target_col_names'] if tc not in df.columns]
        if missing_targets_for_scaling: 
            raise ValueError(f"错误: 一个或多个目标列在准备进行数据缩放前未在DataFrame中找到: {missing_targets_for_scaling}.")
        
        # 11. 数据缩放 (StandardScaler)
        #    - 特征缩放：仅在fit_scalers=True时（即训练新模型时）拟合新的缩放器并保存。
        #    - 目标缩放：对每个目标列单独拟合缩放器并保存（同样仅在fit_scalers=True时）。
        #    - 如果fit_scalers=False（例如预测时），则会加载已保存的缩放器进行transform。
        #      (加载逻辑在AQISystem._ensure_model_loaded_for_use或ModelPredictor._load_artifacts中实现)
        if fit_scalers:
            print("开始拟合和保存数据缩放器...")
            self.feature_scaler = StandardScaler() # 或者可以考虑 RobustScaler 等其他缩放器
            
            # 再次确认用于缩放的特征列确实存在于当前DataFrame中 (df可能因dropna等操作已变化)
            current_features_to_scale = [f_col for f_col in self.all_feature_columns_for_sequence if f_col in df.columns] 
            
            # 在缩放前，对所有选定的特征列进行最终的数值类型检查和强制转换，以防意外的非数值类型。
            for f_col in current_features_to_scale: 
                if not pd.api.types.is_numeric_dtype(df[f_col]):
                    # print(f"特征列 '{f_col}' 在缩放前不是数值类型，尝试转换...") # 日志可能过于频繁
                    df[f_col] = pd.to_numeric(df[f_col], errors='coerce') # 强制转换，无效值变NaN
                    if df[f_col].isnull().any(): # 如果转换后产生NaN
                        # print(f"特征列 '{f_col}' 转换后包含NaN，用0填充。") # 日志可能过于频繁
                        df[f_col] = df[f_col].fillna(0) # 用0填充NaN
                    if not pd.api.types.is_numeric_dtype(df[f_col]): # 再次检查
                         # 如果经过转换和填充后仍然不是数值类型，则抛出错误，指出问题列和一些非数值样本。
                         offending_values = [item for item in df[f_col].unique() if not isinstance(item, (int, float, np.number))]
                         raise ValueError(f"错误: 特征列 '{f_col}' 最终无法转换为数值类型以进行缩放。问题值示例: {offending_values[:5]}")
            
            if not current_features_to_scale: # 如果没有有效的特征列可供缩放
                raise ValueError("错误: 没有有效的特征列可用于拟合特征缩放器。")

            # 对特征列进行拟合和变换
            df[current_features_to_scale] = self.feature_scaler.fit_transform(df[current_features_to_scale])
            # 保存拟合好的特征缩放器
            joblib.dump(self.feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))
            
            # 对每个目标列单独创建、拟合和保存一个缩放器
            self.target_scalers = {} 
            for col_name in self.config['target_col_names']:
                # 确保目标列也是数值类型，并处理可能的NaN（虽然前面已有dropna，但此处作为最后防线）
                if not pd.api.types.is_numeric_dtype(df[col_name]): 
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0) 
                
                scaler = StandardScaler() # 为当前目标列创建一个新的缩放器实例
                # 注意：fit_transform期望2D数组，所以用df[[col_name]]
                df[[col_name]] = scaler.fit_transform(df[[col_name]]) 
                self.target_scalers[col_name] = scaler # 存储拟合好的缩放器
            # 保存包含所有目标缩放器的字典
            joblib.dump(self.target_scalers, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALERS_SAVE_NAME))
            print("特征缩放器和各目标变量的缩放器均已成功拟合和保存。")
        # 注意：如果 fit_scalers 为 False，则此函数不执行缩放器的拟合与保存，
        # 缩放操作将依赖于外部加载的缩放器（通常在预测流程中）。
        # 但此函数仍然返回包含特征工程结果的df，缩放操作会在调用此函数之后，使用已加载的scaler进行。
        # （在当前AQISystem的实现中，预测时的预处理和缩放是在ModelPredictor的_preprocess_input_for_prediction中完成的，
        #  它会加载并使用保存的scalers。）
        return df

    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, trial=None):
        """
        模型训练的核心循环。此函数负责迭代训练数据、计算损失、执行反向传播、
        更新模型参数、在验证集上评估性能、调整学习率以及执行早停策略。
        同时，如果处于Optuna超参数优化试验中，它还会向Optuna报告中间结果并检查是否应剪枝。

        Args:
            model (nn.Module): 待训练的PyTorch模型实例。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            criterion (nn.Module): 损失函数 (例如 nn.MSELoss)。
            optimizer (torch.optim.Optimizer): 优化器 (例如 AdamW)。
            scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器 (例如 ReduceLROnPlateau)。
            epochs (int): 当前训练阶段的总轮数。
            trial (optuna.trial.Trial, optional): Optuna的Trial对象。如果非None，表示当前处于
                                                 超参数优化试验中。默认为None。
        Returns:
            tuple: 包含训练好的模型、训练损失列表、验证损失列表和最佳验证损失（基于主要目标）。
                   (model, train_losses_epoch, val_losses_epoch, best_val_loss)
        """
        best_val_loss = float('inf') # 初始化最佳验证损失为一个极大值
        epochs_no_improve = 0        # 记录验证损失连续未改善的轮数，用于早停
        best_model_state = None      # 用于保存性能最佳时的模型状态字典

        train_losses_epoch = [] # 存储每轮的平均训练损失
        val_losses_epoch = []   # 存储每轮的平均验证损失

        # 获取主要评估目标在多目标输出中的索引，用于早停和Optuna报告
        primary_target_idx = self.config['target_col_names'].index(self.config['primary_target_col_name'])

        for epoch in range(epochs):
            model.train() # 将模型设置为训练模式 (启用Dropout等)
            running_train_loss = 0.0 # 累积当前轮的总训练损失

            # --- 训练阶段 ---
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE) # 数据移至计算设备
                optimizer.zero_grad()  # 清除上一批次的梯度
                outputs = model(X_batch) # 模型前向传播
                loss = criterion(outputs, y_batch) # 计算损失 (所有目标综合损失)
                loss.backward() # 反向传播计算梯度
                optimizer.step() # 更新模型参数
                running_train_loss += loss.item() * X_batch.size(0) # 累加损失 (乘以批大小以得到总损失，而非平均)
            
            # 计算当前轮的平均训练损失
            epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
            train_losses_epoch.append(epoch_train_loss)

            # --- 验证阶段 ---
            model.eval() # 将模型设置为评估模式 (禁用Dropout等)
            running_val_loss = 0.0 # 累积当前轮的总验证损失
            running_primary_target_val_loss = 0.0 # 累积主要评估目标的验证损失

            if len(val_loader.dataset) > 0: # 仅当验证集非空时执行
                with torch.no_grad(): # 评估时不需要计算梯度
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        outputs = model(X_batch)
                        
                        # 计算整体验证损失
                        loss = criterion(outputs, y_batch)
                        running_val_loss += loss.item() * X_batch.size(0)
                        
                        # 单独计算主要评估目标的验证损失 (例如AQI的MSE)
                        # outputs[:, :, primary_target_idx] 提取所有样本、所有horizon步长下，该主要目标的预测值
                        # y_batch[:, :, primary_target_idx] 提取对应的真实值
                        primary_target_loss = criterion(outputs[:, :, primary_target_idx], y_batch[:, :, primary_target_idx])
                        running_primary_target_val_loss += primary_target_loss.item() * X_batch.size(0)
                
                epoch_val_loss = running_val_loss / len(val_loader.dataset) # 平均整体验证损失
                epoch_primary_target_val_loss = running_primary_target_val_loss / len(val_loader.dataset) # 平均主要目标验证损失
            else: # 如果验证集为空，将损失设为无穷大，避免影响判断
                epoch_val_loss = float('inf')
                epoch_primary_target_val_loss = float('inf')
            
            val_losses_epoch.append(epoch_val_loss) # 记录整体验证损失

            # 打印当前轮的训练信息
            current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
            print(f"轮次 [{epoch+1}/{epochs}], 学习率: {current_lr:.7f}, "
                  f"训练损失: {epoch_train_loss:.6f}, "
                  f"验证损失 (综合): {epoch_val_loss:.6f}, "
                  f"验证损失 ({self.config['primary_target_col_name']}): {epoch_primary_target_val_loss:.6f}")

            # --- 学习率调度、Optuna剪枝、早停与模型保存 ---
            if scheduler: # 如果使用了学习率调度器
                scheduler.step(epoch_primary_target_val_loss) # 根据主要目标的验证损失调整学习率

            if trial: # 如果在Optuna试验中
                trial.report(epoch_primary_target_val_loss, epoch) # 向Optuna报告当前轮的主要目标验证损失
                if trial.should_prune(): # 检查Optuna是否建议剪枝（提前终止当前试验）
                    print("Optuna试验被剪枝 (Optuna trial pruned).")
                    raise optuna.exceptions.TrialPruned() # 抛出异常以终止Optuna的当前trial

            # 检查验证集上主要目标的性能是否有改善
            if epoch_primary_target_val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = epoch_primary_target_val_loss # 更新最佳验证损失
                epochs_no_improve = 0 # 重置未改善轮数计数器
                best_model_state = copy.deepcopy(model.state_dict()) # 保存当前最佳模型的状态
                # 如果不是在Optuna试验中（即最终模型训练阶段），则保存模型到文件
                if trial is None: 
                    torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
                    print(f"验证损失 ({self.config['primary_target_col_name']}) 获得改善。模型状态已于轮次 {epoch+1} 保存。")
            else: # 如果验证损失未改善
                epochs_no_improve += 1

            # 早停判断：如果连续多轮验证损失未改善，并且验证集非空
            if epochs_no_improve >= self.config['early_stopping_patience'] and len(val_loader.dataset) > 0 : 
                print(f"早停机制在轮次 {epoch+1} 被触发 (基于 {self.config['primary_target_col_name']} 的验证损失连续 {self.config['early_stopping_patience']} 轮未改善)。")
                if best_model_state: # 如果已保存过最佳状态
                    model.load_state_dict(best_model_state) # 恢复到最佳模型状态
                break # 结束训练循环
        
        # 训练结束后（无论是正常完成还是早停），如果是在最终模型训练阶段且记录了最佳状态，
        # 确保最终模型对象是最佳状态，并再次保存（以防早停发生在最后一轮改善之后）。
        if best_model_state and trial is None: 
             model.load_state_dict(best_model_state) 
             torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
             print(f"训练结束。最终最佳模型状态已保存至 {MODEL_STATE_SAVE_NAME}")
             
        return model, train_losses_epoch, val_losses_epoch, best_val_loss

    def _objective_optuna(self, trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features):
        """
        Optuna超参数优化的目标函数。
        此函数由Optuna在每次试验（trial）时调用，用于定义和训练一个具有特定超参数组合的模型，
        并返回一个评估该组合性能的指标（通常是验证集上的损失）。

        Args:
            trial (optuna.trial.Trial): Optuna的Trial对象，用于建议和记录超参数。
            X_train_np, y_train_np (np.ndarray): 训练集的输入和目标序列。
            X_val_np, y_val_np (np.ndarray): 验证集的输入和目标序列。
            num_input_features (int): 模型输入特征的数量。

        Returns:
            float: 该组超参数在验证集上达到的最佳（最小）主要目标损失。
        """
        # --- 1. 定义超参数搜索空间并由Optuna建议值 ---
        # 学习率 (log尺度搜索，更关注数量级)
        lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True) 
        # Transformer模型内部维度 (d_model)
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512]) 
        # 多头注意力头数 (nhead)，必须能被d_model整除且小于等于d_model
        possible_num_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and d_model >= h]
        if not possible_num_heads: # 如果没有有效的头数选项，则此试验无效，提前剪枝
            raise optuna.exceptions.TrialPruned("对于当前d_model，没有有效的注意力头数选项。")
        num_heads = trial.suggest_categorical('num_heads', possible_num_heads)
        # Transformer编码器层数
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8) 
        # 前馈网络维度相对于d_model的倍数因子 (dim_feedforward = d_model * factor)
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 6) 
        dim_feedforward = d_model * dim_feedforward_factor
        # Dropout比率
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4) 
        # 是否在Transformer层中先应用LayerNorm (norm_first)
        norm_first = trial.suggest_categorical('norm_first', [True, False])
        # AdamW优化器的权重衰减 (weight_decay)，L2正则化的一种形式
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        # ReduceLROnPlateau学习率调度器的参数
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.7) # 学习率衰减因子
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)  # 学习率调度器的耐心轮数

        # --- 2. 构建模型、优化器和损失函数 ---
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
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience) 
        criterion = nn.MSELoss() # 均方误差损失

        # --- 3. 创建数据加载器 ---
        # 注意：num_workers > 0 可能在某些环境（如Jupyter Notebook on Windows）中导致问题，
        #       如果遇到问题，可以尝试设置为0。pin_memory=True可以在GPU可用时加速数据传输。
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True) 
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        
        # 打印当前Optuna试验的参数组合，便于跟踪
        print(f"\nOptuna试验 {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, "
              f"ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.3f}, norm_first={norm_first}, "
              f"wd={weight_decay:.7f}, sch_factor={scheduler_factor:.2f}, sch_patience={scheduler_patience}")

        # --- 4. 执行模型训练（使用当前试验的超参数），并获取最佳验证损失 ---
        # self.config['optuna_epochs'] 是为Optuna单次试验设定的较短训练轮数
        _, _, _, best_val_loss_trial = self._train_model_core(
            model, train_loader, val_loader, criterion, optimizer, scheduler, 
            epochs=self.config['optuna_epochs'], 
            trial=trial # 将trial对象传递给核心训练函数，以便报告和剪枝
        )
        
        return best_val_loss_trial # Optuna将基于此返回值来评估当前超参数组合的优劣

    def run_training_pipeline(self):
        """
        执行完整的模型训练流程。
        该流程包括：数据加载与预处理、数据集划分、Optuna超参数优化、
        使用最佳超参数训练最终模型、评估最终模型性能，并保存所有相关产物。
        """
        print("--- 开始执行模型训练全流程 ---")
        
        # 1. 加载和预处理训练数据，并拟合/保存缩放器
        #    fit_scalers=True 表示这是训练阶段，需要创建并保存新的数据缩放器。
        df_processed = self._load_and_preprocess_data_core(self.config['file_path'], fit_scalers=True)
        
        # 获取模型实际需要的输入特征数量（在预处理后确定）
        num_input_features_for_model = len(self.all_feature_columns_for_sequence) 
        if num_input_features_for_model == 0:
            print("错误: 预处理后没有有效的输入特征列。终止训练。")
            return

        # 2. 从处理后的数据创建输入(X)和目标(y)序列
        X_initial, y_initial = create_sequences(
            df_processed, 
            self.config['look_back'], 
            self.config['horizon'], 
            self.config['target_col_names'], 
            self.all_feature_columns_for_sequence # 使用预处理后确定的特征列
        )
        if X_initial.size == 0 or y_initial.size == 0: 
            print("错误: 创建输入/输出序列后数据为空。可能是原始数据过短或预处理问题。终止训练。")
            return

        # 3. 数据集划分：训练集(70%)、验证集(15%)、测试集(15%)
        total_samples = X_initial.shape[0]
        train_idx_end = int(total_samples * 0.7)
        val_idx_end = int(total_samples * 0.85) # 70% + 15% = 85%
        
        X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]
        X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
        X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:]
        
        print(f"数据集划分完毕: 训练集样本数={X_train_np.shape[0]}, 验证集样本数={X_val_np.shape[0]}, 测试集样本数={X_test_np.shape[0]}")
        if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0: 
            print("错误: 训练集或验证集在划分后为空。无法继续训练。请检查数据量。")
            return

        # 4. 使用Optuna进行超参数优化
        print("\n--- 开始Optuna超参数优化 ---")
        # 创建Optuna Study对象，设置优化方向为最小化（损失），并配置剪枝器和采样器。
        # HyperbandPruner: 一种高效的剪枝策略。
        # TPESampler: Tree-structured Parzen Estimator Sampler，一种常用的贝叶斯优化采样器。
        study = optuna.create_study(
            direction='minimize', 
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=5, # 最小资源（epoch数），少于此则不剪枝
                max_resource=self.config['optuna_epochs'], # 最大资源
                reduction_factor=3 # Hyperband的reduction factor
            ), 
            sampler=optuna.samplers.TPESampler(seed=42) # 设置种子以保证采样过程可复现
        ) 
        # 执行优化过程，调用_objective_optuna函数n_trials次，或直到超时。
        study.optimize(
            lambda trial: self._objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features_for_model), 
            n_trials=self.config['n_optuna_trials'], 
            timeout=3600*6 # 设置一个超时限制（例如6小时），防止优化过程过长
        ) 
        
        best_hyperparams = study.best_params # 获取Optuna找到的最佳超参数组合
        print(f"Optuna超参数优化完成。")
        print(f"最佳试验的验证损失 ({self.config['primary_target_col_name']}): {study.best_value:.6f}")
        print(f"找到的最佳超参数组合: {best_hyperparams}")

        # 5. 使用找到的最佳超参数训练最终模型
        print("\n--- 使用最佳超参数训练最终模型 ---")
        # 从best_hyperparams构建模型架构参数字典
        final_model_arch_params = {
            'd_model': best_hyperparams['d_model'], 
            'nhead': best_hyperparams['num_heads'], 
            'num_encoder_layers': best_hyperparams['num_encoder_layers'],
            'dim_feedforward': best_hyperparams['d_model'] * best_hyperparams['dim_feedforward_factor'], # 注意这里要重新计算
            'dropout': best_hyperparams['dropout_rate'], 
            'norm_first': best_hyperparams['norm_first']
        }
        # 实例化最终模型
        final_model = AQITransformer(
            num_features=num_input_features_for_model, 
            **final_model_arch_params, 
            horizon=self.config['horizon'], 
            num_target_features=len(self.config['target_col_names'])
        ).to(DEVICE)
        
        # 为最终训练创建数据加载器 (可以使用完整的训练集+验证集，或仅训练集，取决于策略。此处用原训练/验证划分)
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        
        # 配置最终模型的优化器和学习率调度器
        final_optimizer = torch.optim.AdamW(
            final_model.parameters(), 
            lr=best_hyperparams['learning_rate'], 
            weight_decay=best_hyperparams.get('weight_decay', 0.0) # 使用get以防optuna未搜索此参数
        )
        final_scheduler = ReduceLROnPlateau(
            final_optimizer, mode='min', 
            factor=best_hyperparams.get('scheduler_factor', 0.5), 
            patience=best_hyperparams.get('scheduler_patience', 7)
        ) 
        criterion = nn.MSELoss() # 损失函数
        
        # 执行最终模型的训练过程
        final_model, train_losses, val_losses, _ = self._train_model_core(
            final_model, final_train_loader, final_val_loader, criterion, final_optimizer, final_scheduler, 
            epochs=self.config['full_train_epochs'] # 使用配置中定义的完整训练轮数
        )
        
        # 绘制并保存最终模型的训练损失曲线
        plot_training_loss(train_losses, val_losses, 
                           os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"), 
                           title_prefix="最终模型")
        print(f"最终模型训练完成。最佳模型状态已保存。")

        # 6. 保存模型配置信息 (包括架构参数、训练设置等)
        model_config_to_save = {
            'model_architecture': final_model_arch_params, 
            'look_back': self.config['look_back'], 
            'horizon': self.config['horizon'],
            'target_col_names': self.config['target_col_names'], 
            'primary_target_col_name': self.config['primary_target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence, # 关键：保存用于序列创建的特征列
            'num_input_features_for_model': num_input_features_for_model,
            'num_target_features': len(self.config['target_col_names']), 
            'optuna_best_params': best_hyperparams, # 保存Optuna找到的最佳参数，供参考
            'enable_iqr_outlier_detection': self.config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION) # 保存IQR开关状态
        } 
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w', encoding='utf-8') as f:
            json.dump(model_config_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False 保证中文正常保存
        print(f"模型配置信息已保存至 {MODEL_CONFIG_SAVE_NAME}。")

        # 7. 在测试集上评估最终模型的性能
        if X_test_np.shape[0] > 0: # 仅当测试集非空时执行评估
            self.evaluate_trained_model(final_model, X_test_np, y_test_np, criterion)
        else:
            print("测试集为空，跳过最终模型评估步骤。")

    def evaluate_trained_model(self, model, X_test_np, y_test_np, criterion):
        """
        在测试集上评估训练好的模型性能，并输出各项指标及可视化结果。

        Args:
            model (nn.Module): 训练好的PyTorch模型。
            X_test_np (np.ndarray): 测试集的输入序列。
            y_test_np (np.ndarray): 测试集的目标序列（真实值）。
            criterion (nn.Module): 损失函数（主要用于参考，实际评估指标更关注MAE, RMSE, R²）。
        """
        print("\n--- 开始在测试集上评估最终模型性能 ---")
        test_dataset = TimeSeriesDataset(X_test_np, y_test_np)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
        
        model.eval() # 设置为评估模式
        all_preds_scaled_list = []   # 存储所有批次的 scaled 预测值
        all_targets_scaled_list = [] # 存储所有批次的 scaled 真实值 (来自y_test_np，本身已是scaled)

        with torch.no_grad(): # 评估时禁用梯度计算
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                outputs_scaled = model(X_batch) # 模型输出的是scaled预测值
                all_preds_scaled_list.append(outputs_scaled.cpu().numpy())
                all_targets_scaled_list.append(y_batch.numpy()) # y_batch已经是NumPy数组，且是scaled
        
        if not all_preds_scaled_list: 
            print("错误: 在测试集上未能生成任何预测。评估中止。")
            return

        # 将所有批次的预测和目标拼接起来
        preds_scaled_np = np.concatenate(all_preds_scaled_list, axis=0)
        targets_scaled_np = np.concatenate(all_targets_scaled_list, axis=0)

        # 将scaled的预测值和目标值反向转换为原始尺度，以便进行有意义的评估
        actual_orig_all_targets = np.zeros_like(targets_scaled_np) # 存储原始尺度的真实值
        predicted_orig_all_targets = np.zeros_like(preds_scaled_np) # 存储原始尺度的预测值

        for i, col_name in enumerate(self.config['target_col_names']):
            scaler = self.target_scalers[col_name] # 获取对应目标的缩放器
            # 反向转换真实值 (targets_scaled_np的第i个目标)
            actual_orig_all_targets[:, :, i] = scaler.inverse_transform(targets_scaled_np[:, :, i])
            # 反向转换预测值 (preds_scaled_np的第i个目标)
            predicted_orig_all_targets[:, :, i] = scaler.inverse_transform(preds_scaled_np[:, :, i])
        
        print("\n各目标污染物在测试集上的评估指标 (原始尺度):")
        for i, col_name in enumerate(self.config['target_col_names']):
            # 提取当前目标的所有时间步的实际值和预测值（展平为1D数组）
            actual_col_flat = actual_orig_all_targets[:, :, i].flatten()
            predicted_col_flat = predicted_orig_all_targets[:, :, i].flatten()
            
            # 计算评估指标
            mae = mean_absolute_error(actual_col_flat, predicted_col_flat)
            rmse = np.sqrt(mean_squared_error(actual_col_flat, predicted_col_flat))
            r2 = r2_score(actual_col_flat, predicted_col_flat)
            
            print(f"  {col_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # 为每个目标绘制实际值与预测值的对比图
            plot_save_prefix = os.path.join(self.config['model_artifacts_dir'], f"final_model_evaluation_test_set") 
            plot_predictions_vs_actual(
                actual_orig_all_targets[:, :, i], # 传递2D数组 (num_sequences, horizon)
                predicted_orig_all_targets[:, :, i], 
                col_name, 
                plot_save_prefix, 
                title_suffix="测试集评估：实际值 vs. 预测值"
            )
        print(f"\n测试集评估图表已保存至目录: {self.config['model_artifacts_dir']}")
        print("\n提示: AQI是基于多种污染物（PM2.5, PM10, SO2, NO2, O3, CO）浓度通过特定算法计算得出的综合性空气质量评价指数。")

class ModelPredictor: 
    """
    该类封装了加载已训练模型并使用其进行预测的逻辑。
    注意：在当前的AQISystem实现中，此类的部分功能（特别是模型和缩放器的加载）
    可能由AQISystem._ensure_model_loaded_for_use()更集中地管理。
    此类主要用于演示一个独立的预测器组件。
    """
    def __init__(self, artifacts_dir):
        """
        初始化ModelPredictor实例。
        Args:
            artifacts_dir (str): 存储已训练模型及其相关组件（配置、缩放器）的目录路径。
        """
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.feature_scaler = None
        self.target_scalers = {}
        self.model_config = None
        self._load_artifacts() # 在实例化时即加载所有必要的组件

    def _load_artifacts(self):
        """
        从指定的artifacts_dir加载模型状态、模型配置以及特征和目标缩放器。
        """
        print(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (ModelPredictor)...")
        
        # 构建各组件文件的完整路径
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME) # 特征缩放器路径
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME) # 目标缩放器路径

        # 检查所有必要文件是否存在
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing_files = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing_files}。请确保模型已正确训练并保存。")
        
        # 加载模型配置 (JSON)
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.model_config = json.load(f)
        
        # 加载特征缩放器和目标缩放器 (Pickle)
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)
        
        # 根据加载的配置实例化模型架构
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'], 
            **self.model_config['model_architecture'], # 解包架构参数
            horizon=self.model_config['horizon'], 
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE)
        
        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval() # 设置为评估模式
        print("模型、配置及缩放器已成功加载 (ModelPredictor)。")

    def _preprocess_input_for_prediction(self, df_raw):
        """
        为预测任务预处理原始输入数据。
        此过程应严格遵循训练数据预处理的步骤，以确保数据分布的一致性。
        包括：时间戳处理、数值转换、特征工程（AQI计算、周期特征、滞后特征）、
        确保所有模型期望的特征列存在，并使用已加载的特征缩放器对特征进行变换。

        Args:
            df_raw (pd.DataFrame): 原始的、未经处理的输入数据DataFrame。
        Returns:
            pd.DataFrame: 经过完整预处理和特征缩放后的DataFrame，可直接用于创建模型输入序列。
        """
        print("开始为预测任务预处理输入数据 (ModelPredictor)...")
        df_processed = df_raw.copy() # 创建副本进行操作

        # 1. 时间戳处理 (与训练时逻辑一致)
        if not isinstance(df_processed.index, pd.DatetimeIndex): 
            if 'Time' in df_processed.columns: 
                df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
                df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                 df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                 df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                 df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: # 尝试第一列
                try:
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                        df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0])
                        df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: 
                    raise ValueError(f"错误: 无法为预测数据自动设置时间索引: {e}。请检查输入数据格式。")
        
        # 2. 数值类型转换和初步NaN填充 (特征列)
        for col in df_processed.columns: 
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce') 
        
        # 对所有非目标列（即特征候补列）的NaN进行填充
        # 注意：目标列在预测时通常不存在于输入数据中，或即使存在也不参与此阶段的填充。
        feature_cols_in_df = [col for col in df_processed.columns if col not in self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)]
        for col in feature_cols_in_df: 
            if df_processed[col].isnull().any(): 
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        # 3. 特征工程 (与训练时逻辑一致)
        #    a. AQI_calculated (如果模型训练时使用了此特征)
        if 'AQI_calculated' in self.model_config['all_feature_columns_for_sequence']:
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi: 
                df_processed = calculate_aqi_from_pollutants(df_processed.copy()) # 传入副本
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                    df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0) # 填充计算失败的AQI
            else: 
                print("警告: 预测数据中缺少计算'AQI_calculated'所需的全部污染物列。将 'AQI_calculated' 特征填充为0。")
                df_processed['AQI_calculated'] = 0 # 如果无法计算，则填充为0
        
        #    b. 周期性特征
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

        #    c. 滞后特征 (基于模型配置中记录的 all_feature_columns_for_sequence)
        lag_features_to_recreate = [f for f in self.model_config['all_feature_columns_for_sequence'] if "_lag_" in f] 
        lag_series_list_pred = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_")
            if len(original_col_name_parts) == 2: # 确保格式正确，如 "PM2.5_lag_1"
                original_col = original_col_name_parts[0]
                lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns: 
                    lag_series_list_pred.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: # 如果原始列在输入数据中不存在，则创建一个全0的滞后特征（或发出更强警告）
                    print(f"警告: 创建滞后特征'{lag_col_name}'所需的原始列'{original_col}'在预测数据中未找到。该滞后特征将填充为0。")
                    lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: # 如果滞后列名格式不符合预期
                print(f"警告: 滞后特征名 '{lag_col_name}' 格式无法解析。该特征将填充为0。")
                lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
        
        if lag_series_list_pred:
             df_processed = pd.concat([df_processed] + lag_series_list_pred, axis=1)

        # 4. 再次填充NaN（滞后特征和周期特征（如果索引非时间）可能引入NaN）
        #    并确保所有模型期望的特征列都存在于df_processed中，不存在则用0填充。
        expected_features_from_config = self.model_config['all_feature_columns_for_sequence'] 
        df_for_scaling = pd.DataFrame(index=df_processed.index) # 用于收集最终特征的DataFrame

        for f_col in expected_features_from_config:
            if f_col not in df_processed.columns: 
                print(f"警告: 模型期望的特征列 '{f_col}' 在预测数据预处理后仍缺失。将创建为全0列。")
                df_for_scaling[f_col] = 0 # 如果特征工程后仍缺失，则添加全0列
            else:
                # 如果特征存在，先填充该列可能存在的NaN
                if df_processed[f_col].isnull().any():
                    df_processed[f_col] = df_processed[f_col].ffill().bfill().fillna(0)
                df_for_scaling[f_col] = df_processed[f_col] 
        
        # 5. 特征缩放：使用已加载的feature_scaler对选定的特征列进行transform
        #    在缩放前，最后一次确保所有待缩放特征列都是数值类型。
        for f_col in expected_features_from_config: 
             df_for_scaling[f_col] = pd.to_numeric(df_for_scaling[f_col], errors='coerce').fillna(0) # 再次强制转换和填充
             if not pd.api.types.is_numeric_dtype(df_for_scaling[f_col]):
                # 如果此时仍有非数值列，说明数据问题严重，可能需要抛出错误
                try: 
                    df_for_scaling[f_col] = df_for_scaling[f_col].astype(float)
                except ValueError as e_astype:
                    offending_values = [item for item in df_for_scaling[f_col].unique() if not isinstance(item, (int, float, np.number))]
                    raise ValueError(f"错误: 无法将预测数据中的特征列 '{f_col}' 转换为浮点数以进行缩放。问题值示例: {offending_values[:5]}. 原错误: {e_astype}")

        # 执行特征缩放，注意只选择模型配置中记录的特征列进行变换
        df_for_scaling_transformed = self.feature_scaler.transform(df_for_scaling[expected_features_from_config]) 
        # 将缩放后的NumPy数组转回DataFrame，保持原始索引和列名
        df_scaled_features = pd.DataFrame(df_for_scaling_transformed, columns=expected_features_from_config, index=df_for_scaling.index)
        
        print("预测输入数据预处理完成。")
        return df_scaled_features

    def predict(self, input_data_path_or_df):
        """
        使用加载的模型对输入数据进行预测。

        Args:
            input_data_path_or_df (str or pd.DataFrame): 
                可以是包含预测所需历史数据的CSV/Excel文件路径，
                也可以是已经加载到内存中的Pandas DataFrame。
                该数据应至少包含模型回溯窗口 (look_back) 长度的连续记录。
        Returns:
            tuple: (predicted_values_original_scale, last_known_timestamp)
                   - predicted_values_original_scale (np.ndarray or None): 
                       形状为 (horizon, num_target_features) 的NumPy数组，包含原始尺度的预测值。
                       如果预测失败，则为None。
                   - last_known_timestamp (pd.Timestamp or None):
                       从输入数据中解析得到的最后一个有效时间戳。如果无法确定，则为None。
                       此时间戳用于后续生成预测结果的时间标签。
        Raises:
            ValueError: 如果输入数据不符合要求（如类型错误、长度不足等）。
            FileNotFoundError: 如果输入的是文件路径但文件未找到。
        """
        # 1. 加载原始数据 (如果是文件路径)
        if isinstance(input_data_path_or_df, str):
            try: 
                df_raw = pd.read_csv(input_data_path_or_df)
            except Exception: # 更通用的异常捕获
                try:
                    df_raw = pd.read_excel(input_data_path_or_df)
                except FileNotFoundError:
                    raise FileNotFoundError(f"错误: 预测数据文件 '{input_data_path_or_df}' 未找到。")
                except Exception as e_excel:
                    raise ValueError(f"错误: 读取Excel预测数据文件 '{input_data_path_or_df}' 失败: {e_excel}")
        elif isinstance(input_data_path_or_df, pd.DataFrame):
            df_raw = input_data_path_or_df.copy() # 使用副本
        else:
            raise ValueError("错误: 输入数据必须是CSV/Excel文件路径或Pandas DataFrame。")
        
        # 2. 尝试确定输入数据中的最后一个已知时间戳，用于后续生成预测的时间标签
        last_known_timestamp = None 
        if isinstance(df_raw.index, pd.DatetimeIndex) and not df_raw.empty: 
            last_known_timestamp = df_raw.index[-1]
        elif not df_raw.empty: # 如果不是DatetimeIndex，但数据非空，尝试从列中解析
            time_cols_to_check = ['Time', 'timestamp', 'Datetime'] # 常见的可能时间列名
            for tc in time_cols_to_check:
                if tc in df_raw.columns: 
                    try: last_known_timestamp = pd.to_datetime(df_raw[tc].iloc[-1]); break
                    except: pass # 解析失败则尝试下一个
            if last_known_timestamp is None and 'date' in df_raw.columns and 'hour' in df_raw.columns: # 尝试 date/hour 组合
                try:
                    date_val = str(df_raw['date'].iloc[-1])
                    hour_val = str(int(df_raw['hour'].iloc[-1])).zfill(2) # 确保hour是两位数
                    last_dt_str = date_val + hour_val
                    last_known_timestamp = pd.to_datetime(last_dt_str, format='%Y%m%d%H')
                except: pass
            elif last_known_timestamp is None: # 最后尝试第一列
                 try:
                     if not pd.api.types.is_numeric_dtype(df_raw.iloc[:,0]): # 避免将纯数字列误认为时间
                         last_known_timestamp = pd.to_datetime(df_raw.iloc[-1, 0])
                 except: pass # 所有尝试均失败
        
        if last_known_timestamp is None: 
            print("警告: 无法从输入数据中可靠地确定最后一个时间戳。预测结果的时间标签可能不准确。")

        # 3. 对原始数据进行预处理和特征缩放
        df_processed_features = self._preprocess_input_for_prediction(df_raw.copy()) # 传入副本
        
        # 4. 检查处理后的数据长度是否满足模型回溯窗口要求
        if len(df_processed_features) < self.model_config['look_back']:
            raise ValueError(f"错误: 经过预处理后的预测输入数据长度为 {len(df_processed_features)}，"
                             f"不足以满足模型所需的回溯窗口长度 {self.model_config['look_back']}。")
        
        # 5. 准备模型输入序列 (X_pred)
        #    只选择模型配置中记录的 all_feature_columns_for_sequence 进行输入
        model_input_feature_names = self.model_config['all_feature_columns_for_sequence']
        # 从处理后的数据末尾提取look_back长度的序列，并确保只包含模型期望的特征列
        # 先筛选df_processed_features中实际存在的、且在model_input_feature_names中的列
        available_model_features = [col for col in model_input_feature_names if col in df_processed_features.columns]
        if len(available_model_features) != len(model_input_feature_names):
            missing_for_pred = set(model_input_feature_names) - set(available_model_features)
            print(f"警告: 预处理后，部分模型期望的特征列在最终输入数据中缺失: {missing_for_pred}。这些列可能已被填充为0，但请核实。")
        
        # 使用实际可用的特征列来构建序列，以避免KeyError
        last_input_sequence_df = df_processed_features[available_model_features].iloc[-self.model_config['look_back']:]
        
        X_pred_np = np.array([last_input_sequence_df.values]) # 增加一个批次维度 (1, look_back, num_features)
        if X_pred_np.size == 0: 
            print("错误: 无法创建用于预测的输入序列 (X_pred_np为空)。")
            return None, last_known_timestamp # 返回None表示预测失败

        # 6. 模型预测
        X_pred_torch = torch.from_numpy(X_pred_np).float().to(DEVICE) # 转为PyTorch张量并移至设备
        with torch.no_grad(): # 预测时无需梯度计算
            predictions_scaled = self.model(X_pred_torch) # 模型输出的是缩放后的预测值
        
        # 7. 将预测结果反向转换为原始尺度
        # predictions_scaled 形状: (1, horizon, num_targets)
        # 我们需要 (horizon, num_targets)
        predictions_scaled_np_slice = predictions_scaled.cpu().numpy()[0, :, :] 
        
        predictions_original_all_targets = np.zeros_like(predictions_scaled_np_slice) # 初始化存储原始尺度预测值的数组
        for i, col_name in enumerate(self.model_config['target_col_names']):
            scaler = self.target_scalers[col_name] # 获取对应目标的缩放器
            pred_col_scaled_reshaped = predictions_scaled_np_slice[:, i].reshape(-1, 1) # (horizon, 1)
            predictions_original_all_targets[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten() # (horizon,)
        
        print(f"已成功生成 {self.model_config['horizon']} 个时间步的预测。")
        return predictions_original_all_targets, last_known_timestamp

class AQISystem:
    """
    AQI预测与异常检测系统的主控制类。
    该类整合了模型训练、加载、预测以及新增的异常检测功能，
    为用户提供一个统一的操作接口。
    """
    def __init__(self, artifacts_dir=MODEL_ARTIFACTS_DIR, config_overrides=None):
        """
        初始化AQISystem实例。
        Args:
            artifacts_dir (str, optional): 存储或加载模型工件（权重、配置、缩放器等）的根目录。
                                           默认为全局定义的MODEL_ARTIFACTS_DIR。
            config_overrides (dict, optional): 一个字典，用于覆盖部分或全部默认配置参数。
                                               默认为None，即使用所有默认配置。
        """
        self.artifacts_dir = artifacts_dir
        self.config = self._load_default_config() # 加载基础配置
        if config_overrides: # 如果提供了覆盖配置，则更新
            self.config.update(config_overrides)
        
        # 确保配置中的模型工件目录与实例的artifacts_dir一致
        self.config['model_artifacts_dir'] = self.artifacts_dir 
        os.makedirs(self.artifacts_dir, exist_ok=True) # 确保工件目录存在
        set_seed() # 设置全局随机种子，保证可复现性

        self.trainer = None # ModelTrainer实例，在需要训练时创建
        self.predictor_instance = None # ModelPredictor实例，在需要预测时创建或复用
        
        # 用于存储已加载的模型及其相关组件，避免重复加载
        self.model_config = None
        self.feature_scaler = None
        self.target_scalers = None
        self.model = None
        self.all_feature_columns_for_sequence = None # 关键：从模型配置中加载，用于确保预处理一致性

    def _load_default_config(self):
        """加载系统的默认配置参数。"""
        return {
            'file_path': DEFAULT_FILE_PATH, 
            'look_back': DEFAULT_LOOK_BACK, 
            'horizon': DEFAULT_HORIZON,
            'target_col_names': DEFAULT_TARGET_COL_NAMES, 
            'primary_target_col_name': DEFAULT_PRIMARY_TARGET_COL_NAME,
            'batch_size': DEFAULT_BATCH_SIZE, 
            'model_artifacts_dir': self.artifacts_dir, # 确保与实例的artifacts_dir同步
            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS, 
            'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS, 
            'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA, 
            'anomaly_threshold_factor': DEFAULT_ANOMALY_THRESHOLD_FACTOR,
            'enable_iqr_outlier_detection': DEFAULT_ENABLE_IQR_OUTLIER_DETECTION # 新增：IQR开关配置
        }

    def _ensure_model_loaded_for_use(self):
        """
        确保模型、相关的缩放器和配置信息已从文件加载到AQISystem实例的属性中。
        如果尚未加载，则执行加载操作。此方法旨在避免重复加载，提高效率。
        主要在执行预测或异常检测前调用。
        """
        # 如果所有关键组件都已加载，则直接返回
        if self.model is not None and self.model_config is not None and \
           self.feature_scaler is not None and self.target_scalers is not None and \
           self.all_feature_columns_for_sequence is not None:
            return 

        print(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (AQISystem集中管理)...")
        
        # 构建各组件文件的完整路径
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)

        # 检查所有必要文件是否存在
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing}。请确保模型已成功训练并保存，或指定的目录正确。")

        # 1. 加载模型配置 (JSON)
        with open(config_path, 'r', encoding='utf-8') as f: 
            self.model_config = json.load(f)
        
        # 2. 根据加载的模型配置，更新AQISystem实例的部分配置，以确保一致性
        #    例如，look_back, horizon, target_col_names 等应与训练时使用的配置一致。
        self.config['look_back'] = self.model_config.get('look_back', self.config['look_back'])
        self.config['horizon'] = self.model_config.get('horizon', self.config['horizon'])
        self.config['target_col_names'] = self.model_config.get('target_col_names', self.config['target_col_names'])
        self.config['primary_target_col_name'] = self.model_config.get('primary_target_col_name', self.config['primary_target_col_name'])
        # 从模型配置中加载IQR开关状态，如果模型配置中没有，则使用全局默认值
        self.config['enable_iqr_outlier_detection'] = self.model_config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION)
        
        # 关键：加载训练时确定的用于创建序列的特征列列表
        self.all_feature_columns_for_sequence = self.model_config.get('all_feature_columns_for_sequence')
        if self.all_feature_columns_for_sequence is None:
            raise ValueError("错误: 加载的模型配置 (model_config.json) 中未找到 'all_feature_columns_for_sequence'。此信息对于数据预处理至关重要。")

        # 3. 加载特征缩放器和目标缩放器 (Pickle)
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)

        # 4. 根据加载的配置实例化模型架构并加载权重
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'],
            **self.model_config['model_architecture'], # 解包模型架构参数
            horizon=self.model_config['horizon'],
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE)) # map_location确保在无GPU环境也能加载
        self.model.eval() # 设置为评估模式
        
        print("模型、配置及缩放器已成功加载并准备就绪 (AQISystem集中管理)。")

    def train_new_model(self, train_data_path, enable_iqr_detection_override=None):
        """
        启动并管理新的模型训练流程。
        Args:
            train_data_path (str): 训练数据文件的路径。
            enable_iqr_detection_override (bool, optional): 用于覆盖配置中IQR开关状态的值。
        """
        self.config['file_path'] = train_data_path # 更新配置中的训练文件路径
        if enable_iqr_detection_override is not None:
            self.config['enable_iqr_outlier_detection'] = enable_iqr_detection_override
            print(f"信息: IQR异常值检测功能已通过命令行设置为: {'启用' if enable_iqr_detection_override else '禁用'}")

        self.trainer = ModelTrainer(self.config) # 创建ModelTrainer实例，传入当前配置
        try:
            self.trainer.run_training_pipeline() # 执行完整的训练流程
            # 训练完成后，主动将新训练的模型及其组件加载到AQISystem的属性中，
            # 以便后续可以直接使用（例如，如果用户训练后立刻想预测或检测异常）。
            print("模型训练完成。尝试加载新训练的模型组件到当前AQISystem实例...")
            self._ensure_model_loaded_for_use() 
        except Exception as e:
            print(f"模型训练过程中发生严重错误: {e}")
            import traceback; traceback.print_exc() # 打印详细的错误堆栈信息

    def predict_with_existing_model(self, input_data_path_or_df):
        """
        使用已加载（或按需加载）的现有模型对新数据进行预测。
        Args:
            input_data_path_or_df (str or pd.DataFrame): 预测输入数据（文件路径或DataFrame）。
        """
        try:
            self._ensure_model_loaded_for_use() # 确保模型等核心组件已加载到AQISystem实例

            # 理想情况下，预测逻辑应直接使用self.model, self.feature_scaler等AQISystem的属性，
            # 而不是每次都重新实例化ModelPredictor。
            # 当前为了保持ModelPredictor的相对独立性，暂时仍通过它进行。
            # 但注意，ModelPredictor在__init__中会自行加载文件，这可能与_ensure_model_loaded_for_use的加载有冗余。
            # 一个优化方向是修改ModelPredictor，使其可以接收已加载的组件作为参数。
            
            # 简单复用ModelPredictor实例，如果目录未变则不重新创建
            if self.predictor_instance is None or self.predictor_instance.artifacts_dir != self.artifacts_dir:
                 self.predictor_instance = ModelPredictor(self.artifacts_dir)
            
            # 调用ModelPredictor的预测方法
            predicted_values, last_timestamp = self.predictor_instance.predict(input_data_path_or_df)

            # --- 处理和展示预测结果 ---
            if predicted_values is not None: 
                # 获取预测的时间范围 (horizon) 和目标列名，应从 self.predictor_instance.model_config 获取，
                # 因为这是与实际执行预测的模型相关联的配置。
                horizon_from_pred_config = self.predictor_instance.model_config.get('horizon', self.config['horizon'])
                target_names_from_pred_config = self.predictor_instance.model_config.get('target_col_names', self.config['target_col_names'])

                print(f"\n模型成功生成了未来 {horizon_from_pred_config} 小时的各项指标预测值 (原始尺度):")
                # 打印部分预测结果（例如前5小时，或所有，如果horizon较短）
                for h in range(min(5, horizon_from_pred_config)): 
                    hour_str = f"  未来第 {h+1} 小时: "
                    for t_idx, t_name in enumerate(target_names_from_pred_config): 
                        val_to_print = predicted_values[h, t_idx]
                        if t_name == 'CO': # CO通常保留两位小数
                            hour_str += f"{t_name}={val_to_print:.2f} "
                        else: # 其他指标通常取整
                            hour_str += f"{t_name}={np.round(val_to_print).astype(int)} "
                    print(hour_str)
                
                # 如果成功解析了最后一个已知时间戳，则可以生成带时间标签的CSV输出
                if last_timestamp is not None: 
                    save_pred = input("是否将详细预测结果保存到CSV文件? (y/n): ").strip().lower() 
                    if save_pred.startswith('y'): 
                        pred_save_path = os.path.join(self.artifacts_dir, "predictions_output_adv.csv") 
                        # 生成未来预测时间戳序列
                        future_timestamps = pd.date_range(
                            start=last_timestamp + pd.Timedelta(hours=1), 
                            periods=horizon_from_pred_config, 
                            freq='H' # 每小时一个预测点
                        )
                        output_data = {'date': future_timestamps.strftime('%Y%m%d'), 'hour': future_timestamps.hour}
                        
                        # 整理预测数据到输出字典
                        for t_idx, t_name in enumerate(target_names_from_pred_config): 
                            val_col = predicted_values[:, t_idx] # 获取该目标的所有horizon步的预测
                            if t_name == 'CO': 
                                output_data[t_name] = np.round(val_col, 2)
                            else: 
                                output_data[t_name] = np.round(val_col).astype(int)
                                
                        # 定义期望的输出列顺序，并确保所有列都存在（不存在的用NaN填充）
                        requested_output_columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
                        final_output_df_cols = {}
                        for col in requested_output_columns:
                            final_output_df_cols[col] = output_data.get(col, [np.nan] * horizon_from_pred_config)
                        output_df = pd.DataFrame(final_output_df_cols)[requested_output_columns] # 按指定顺序创建DataFrame
                        
                        output_df.to_csv(pred_save_path, index=False, encoding='utf-8-sig') # 保存CSV，utf-8-sig支持Excel中文
                        print(f"预测结果已保存至: {pred_save_path}")
                        
                        # 打印关于CSV内容的额外说明
                        predicted_cols_in_output = [col for col in target_names_from_pred_config if col in requested_output_columns]
                        nan_cols_in_output = [col for col in requested_output_columns if col not in ['date', 'hour'] and col not in predicted_cols_in_output]
                        print(f"\n提示: CSV文件中实际包含预测数据的列为: {', '.join(predicted_cols_in_output)}。")
                        if nan_cols_in_output: 
                            print(f"以下在标准输出列中但未被当前模型预测的列，在CSV中已填充为NaN: {', '.join(nan_cols_in_output)}。")
                    else: 
                        print("预测结果未保存。")
                else: # 如果last_timestamp为None
                    print("由于无法从输入数据中确定最后一个有效时间戳，预测结果未关联具体日期，也无法保存为带时间戳的CSV文件。")
                    print("原始预测值数组 (形状: horizon, num_targets):")
                    print(np.round(predicted_values,2)) # 打印原始数组供用户参考

            elif predicted_values is None: # 如果predict方法返回None
                print("模型未能成功生成预测。请检查日志或输入数据。")
        
        except FileNotFoundError as e:
             print(f"文件未找到错误 (预测流程): {e}。请检查模型工件目录和输入文件路径。")
        except ValueError as e:
             print(f"值错误 (预测流程): {e}")
        except Exception as e:
             print(f"预测过程中发生未知错误: {e}")
             import traceback; traceback.print_exc()


    def _preprocess_input_for_anomaly(self, df_raw):
        """
        为异常检测任务专门预处理原始输入数据。
        与预测预处理类似，但关键区别在于：
        - 目标列的值需要保留其原始（但经过清洗和数值转换的）尺度，因为它们将作为比较的基准。
        - 特征列仍然需要进行与训练时一致的特征工程和缩放。
        
        Args:
            df_raw (pd.DataFrame): 原始输入数据DataFrame。
        Returns:
            pd.DataFrame: 一个DataFrame，其中包含了经过缩放的特征列，
                          以及保持原始尺度（但已清洗）的目标列。
                          此DataFrame将用于后续的create_sequences调用。
        """
        print("开始为异常检测任务预处理输入数据 (AQISystem)...")
        df_processed = df_raw.copy() # 创建副本

        # 1. 时间戳处理 (与训练/预测时逻辑一致)
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            # (此处省略了与 predict_with_existing_model 中相同的详细时间戳解析逻辑，假设其已正确实现)
            if 'Time' in df_processed.columns:
                df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
                df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: 
                try: # 尝试第一列
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                         df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0])
                         df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e: 
                    raise ValueError(f"错误: 无法为异常检测数据自动设置时间索引: {e}。")

        # 2. 转换为数值类型
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 3. 处理NaN值：
        #    - 特征列：采用与训练时类似的填充策略 (ffill -> bfill -> 0)
        #    - 目标列：对于异常检测，我们需要真实的观测值。如果目标列中存在NaN，
        #              通常表示数据缺失，这些点可能无法直接用于与模型预测比较。
        #              此处简单用0填充，并在日志中警告。更复杂的处理可能包括忽略这些点或采用特定插值。
        
        # 使用 self.model_config (应已通过 _ensure_model_loaded_for_use 加载) 中的目标列名
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        feature_cols_in_df_anomaly = [col for col in df_processed.columns if col not in model_target_cols]
        
        for col in feature_cols_in_df_anomaly: # 填充特征列的NaN
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)
        
        for tc in model_target_cols: # 处理目标列的NaN
            if tc in df_processed.columns:
                if df_processed[tc].isnull().any():
                    print(f"警告: 目标列 '{tc}' 在用于异常检测的数据中包含NaN值。这些NaN值将被填充为0，可能影响异常检测的准确性。")
                    df_processed[tc] = df_processed[tc].fillna(0)
            else: # 如果数据中完全缺少某个模型训练时定义的目标列
                print(f"严重警告: 模型训练时定义的目标列 '{tc}' 在提供的异常检测数据中完全缺失。将为此列创建全0数据，但这很可能导致无效的异常检测结果。")
                df_processed[tc] = 0 # 创建全0列，以保证后续代码流程不因缺列而中断

        # 4. 特征工程 (AQI_calculated, 周期特征, 滞后特征)
        #    与训练/预测时的逻辑保持一致，使用 self.all_feature_columns_for_sequence (来自加载的模型配置)
        #    来确定需要哪些特征。
        
        #    a. AQI_calculated
        if 'AQI_calculated' in self.all_feature_columns_for_sequence: 
            pollutant_cols_for_aqi_calc = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
            can_calc_aqi = all(p_col in df_processed.columns for p_col in pollutant_cols_for_aqi_calc)
            if can_calc_aqi:
                df_processed = calculate_aqi_from_pollutants(df_processed.copy()) # 传入副本
                if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                     df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            else:
                print("警告: 异常检测数据缺少计算'AQI_calculated'所需的全部或部分污染物列。'AQI_calculated'特征将填充为0。")
                df_processed['AQI_calculated'] = 0
        
        #    b. 周期性特征
        new_cyclical_features = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            idx = df_processed.index
            # (省略了周期特征的具体计算，假设与训练/预测时一致)
            new_cyclical_features['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
            df_processed = pd.concat([df_processed, new_cyclical_features], axis=1)
        else:
            # 如果模型依赖周期特征但此处无法创建，应有警告
            if any(cyc_feat in self.all_feature_columns_for_sequence for cyc_feat in ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']):
                 print("警告: 异常检测数据的索引非DatetimeIndex，无法创建周期性时间特征。若模型依赖这些特征，结果可能不准确。")


        #    c. 滞后特征
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

        # 再次填充NaN (滞后特征可能引入NaN，特别是在数据前端)
        for col in df_processed.columns: # 遍历所有列
            if df_processed[col].isnull().any(): 
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)

        # 5. 准备特征进行缩放，并构建最终用于create_sequences的DataFrame
        #    该DataFrame将包含：
        #    - 经过缩放的特征列 (使用self.feature_scaler)
        #    - 原始尺度（但已清洗）的目标列
        
        df_features_for_scaling = pd.DataFrame(index=df_processed.index) # 存储待缩放的特征
        # 从df_processed中提取模型期望的特征列
        for f_col in self.all_feature_columns_for_sequence: 
            if f_col not in df_processed.columns:
                print(f"警告: 模型期望的特征列 '{f_col}' 在异常检测数据特征工程后仍缺失。将创建为全0列用于缩放。")
                df_features_for_scaling[f_col] = 0 # 如果缺失，用0填充
            else:
                df_features_for_scaling[f_col] = df_processed[f_col]
        
        # 确保所有待缩放特征列是数值类型
        for f_col in self.all_feature_columns_for_sequence: 
             df_features_for_scaling[f_col] = pd.to_numeric(df_features_for_scaling[f_col], errors='coerce').fillna(0)
             if not pd.api.types.is_numeric_dtype(df_features_for_scaling[f_col]):
                try: 
                    df_features_for_scaling[f_col] = df_features_for_scaling[f_col].astype(float)
                except ValueError: 
                    raise ValueError(f"错误: 无法将异常检测数据中的特征列 '{f_col}' 转换为浮点数以进行缩放。")
        
        # 6. 使用已加载的特征缩放器 (self.feature_scaler) 对特征进行transform
        #    确保只对模型配置中定义的 all_feature_columns_for_sequence 进行操作
        df_scaled_features_part = self.feature_scaler.transform(df_features_for_scaling[self.all_feature_columns_for_sequence])
        df_scaled_features_df = pd.DataFrame(df_scaled_features_part, columns=self.all_feature_columns_for_sequence, index=df_processed.index)

        # 7. 组合：将缩放后的特征与原始尺度的目标列合并，形成最终用于create_sequences的DataFrame
        df_final_for_sequences = df_scaled_features_df.copy()
        for tc in model_target_cols: # 使用从模型配置中获取的目标列名
            if tc in df_processed.columns: # 目标列应该存在于df_processed中（可能已用0填充）
                df_final_for_sequences[tc] = df_processed[tc] # 添加原始尺度（但已清洗）的目标值
            else: 
                 # 此情况理论上不应发生，因为前面已处理过目标列缺失的问题
                 print(f"严重内部错误: 目标列 '{tc}' 在合并阶段未找到于df_processed。将填充为0。")
                 df_final_for_sequences[tc] = 0 
        
        # 最后一次全面检查NaN，确保传递给create_sequences的数据是完全干净的
        # 检查的列包括所有特征列和所有目标列
        cols_for_sequence_creation = self.all_feature_columns_for_sequence + model_target_cols
        for col in cols_for_sequence_creation:
            # 确保只检查实际存在于df_final_for_sequences中的列
            if col in df_final_for_sequences.columns and df_final_for_sequences[col].isnull().any():
                print(f"警告: 列 '{col}' 在准备创建序列用于异常检测前仍包含NaN值。将用0填充。")
                df_final_for_sequences[col] = df_final_for_sequences[col].fillna(0)
        
        if df_final_for_sequences.empty:
            raise ValueError("错误: 数据在为异常检测任务预处理完毕后变为空。请检查原始数据或预处理步骤。")
            
        print("异常检测输入数据预处理完成。")
        return df_final_for_sequences


    def detect_anomalies(self, data_path_or_df, threshold_factor=None):
        """
        使用已加载的模型检测输入数据中的异常点。
        异常是基于模型对下一步的预测与实际观测值之间的差异来定义的。

        Args:
            data_path_or_df (str or pd.DataFrame): 包含待检测数据的CSV/Excel文件路径或DataFrame。
                                                 此数据应包含模型训练时使用的所有特征的原始值，
                                                 以及所有目标变量的实际观测值。
            threshold_factor (float, optional): 用于定义异常阈值的因子。阈值通常计算为：
                                                mean_absolute_error + threshold_factor * std_dev_of_absolute_error。
                                                如果为None，则使用配置中的默认值。
        Returns:
            dict: 一个字典，键为目标污染物名称，值为包含该污染物异常检测结果的报告。
                  每个报告包含：异常点数量、使用的阈值、异常点的时间戳列表等。
                  如果无法执行检测（例如数据不足），则可能返回空字典。
        """
        self._ensure_model_loaded_for_use() # 确保模型、配置和缩放器已加载

        # 确定异常检测阈值因子
        current_threshold_factor = threshold_factor if threshold_factor is not None \
                                   else self.config.get('anomaly_threshold_factor', DEFAULT_ANOMALY_THRESHOLD_FACTOR)
        print(f"开始执行异常数据检测流程，使用的阈值因子为: {current_threshold_factor} ...")

        # 1. 加载原始数据
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

        # 2. 对数据进行异常检测专用的预处理
        #    返回的 df_for_sequences 包含：缩放后的特征 + 原始尺度的目标值
        df_for_sequences = self._preprocess_input_for_anomaly(df_raw_full.copy())

        # 3. 校验预处理后的数据是否包含所有必要的目标列（原始尺度）
        #    这些目标列将用于与模型预测进行比较。
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        missing_targets_in_processed_df = [tc for tc in model_target_cols 
                                           if tc not in df_for_sequences.columns or df_for_sequences[tc].isnull().all()]
        if missing_targets_in_processed_df:
            raise ValueError(f"错误: 异常检测数据在预处理后，缺少一个或多个有效的实际目标观测列: {missing_targets_in_processed_df}。无法进行异常比较。")

        # 4. 创建用于异常检测的序列
        #    - horizon=1: 因为我们通常比较模型对紧邻下一步的预测与实际值。
        #    - is_predict=False: 因为我们需要y_actual_anomaly_np（实际观测值）进行比较。
        X_anomaly_np, y_actual_anomaly_np = create_sequences(
            df_for_sequences, 
            look_back=self.model_config['look_back'], # 使用加载的模型配置中的look_back
            horizon=1, 
            target_col_names=model_target_cols, 
            feature_cols=self.all_feature_columns_for_sequence, # 使用加载的模型配置中的特征列列表
            is_predict=False
        )

        if X_anomaly_np.size == 0: # 如果无法生成任何序列
            print("警告: 无法为异常检测创建有效的输入/输出序列。可能是数据量过少或预处理后数据不满足要求。异常检测中止。")
            return {} # 返回空字典表示没有检测结果

        # 5. 获取模型对X_anomaly_np的预测 (这些预测是缩放后的)
        self.model.eval() # 确保模型处于评估模式
        all_predictions_scaled_list = []
        
        # 为防止大数据一次性送入模型导致内存问题，可以分批处理
        anomaly_dataset = TensorDataset(torch.from_numpy(X_anomaly_np).float())
        batch_size_for_anomaly = self.config.get('batch_size', DEFAULT_BATCH_SIZE) 
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size_for_anomaly, shuffle=False)
        
        with torch.no_grad(): # 预测时无需梯度
            for x_batch_tuple in anomaly_loader: # DataLoader返回的是元组 (X_batch,)
                x_batch = x_batch_tuple[0].to(DEVICE)
                # 模型输出的是整个horizon的预测，即使我们只关心第一步
                outputs_scaled_batch = self.model(x_batch) 
                all_predictions_scaled_list.append(outputs_scaled_batch.cpu().numpy())
        
        if not all_predictions_scaled_list:
            print("错误: 模型未能对异常检测数据生成任何预测输出。异常检测中止。")
            return {}
            
        predictions_scaled_full_horizon = np.concatenate(all_predictions_scaled_list, axis=0)
        # 我们只关心对下一步(horizon=0)的预测
        predictions_scaled_one_step = predictions_scaled_full_horizon[:, 0, :] # 形状: (num_sequences, num_targets)

        # 6. 将缩放后的单步预测值反向转换为原始尺度
        predicted_original_one_step = np.zeros_like(predictions_scaled_one_step)
        for i, col_name in enumerate(model_target_cols):
            scaler = self.target_scalers[col_name] # 获取对应目标的缩放器
            # reshape to (num_sequences, 1) for inverse_transform
            pred_col_scaled_reshaped = predictions_scaled_one_step[:, i].reshape(-1, 1) 
            predicted_original_one_step[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten()
        
        # 7. y_actual_anomaly_np 已经是原始尺度 (来自df_for_sequences的目标列)，
        #    其形状是 (num_sequences, 1, num_targets)。我们需要提取出 (num_sequences, num_targets)。
        actual_original_one_step = y_actual_anomaly_np[:, 0, :]

        # 8. 计算预测误差 (此处使用绝对误差 MAE)
        errors = np.abs(actual_original_one_step - predicted_original_one_step) 

        # 9. 为每个目标污染物生成异常报告
        anomaly_reports = {}
        print("\n--- 异常数据检测详细报告 ---")
        num_sequences_generated = X_anomaly_np.shape[0]
        
        # 确定用于绘图和报告的时间戳/索引。
        # 这些时间戳对应于 y_actual_anomaly_np (即 actual_original_one_step) 中的每一个数据点。
        # create_sequences 生成的 y_actual_anomaly_np[k] 对应于 df_for_sequences 中的第 look_back + k 个时间点的数据。
        # 因此，时间戳应从 df_for_sequences.index[look_back] 开始，取 num_sequences_generated 个。
        if len(df_for_sequences.index) >= self.model_config['look_back'] + num_sequences_generated:
            base_timestamps_for_y = df_for_sequences.index[self.model_config['look_back'] : self.model_config['look_back'] + num_sequences_generated]
        else: # 如果df_for_sequences的长度不足，这理论上不应发生，因为create_sequences会处理
            print("警告: 用于异常检测的时间戳序列长度不足，可能导致绘图或报告的时间不准确。将使用简单范围索引。")
            base_timestamps_for_y = pd.RangeIndex(start=0, stop=num_sequences_generated, step=1)


        for i, col_name in enumerate(model_target_cols):
            col_errors = errors[:, i] # 当前目标的所有序列的误差
            
            if len(col_errors) == 0: 
                print(f"\n目标污染物: {col_name} - 无有效误差数据可供分析。")
                anomaly_reports[col_name] = {'count': 0, 'threshold': np.nan, 'timestamps': [], 'details': []}
                continue

            # 计算误差的均值和标准差，用于动态设定阈值
            mean_error = np.mean(col_errors)
            std_error = np.std(col_errors)
            
            # 定义异常阈值：均值 + N倍标准差。增加一个极小值防止std_error为0时阈值不变。
            threshold = mean_error + current_threshold_factor * std_error if std_error > 1e-9 else mean_error + 1e-9 
            
            # 识别超出阈值的异常点
            anomaly_flags_col = col_errors > threshold
            # 获取这些异常点在 errors 数组（也是在 actual_original_one_step 和 predicted_original_one_step 中）的索引
            current_col_anomaly_indices_relative = np.where(anomaly_flags_col)[0] 

            # 将这些相对索引映射回原始数据的时间戳/索引
            anomaly_timestamps_col = [base_timestamps_for_y[k] for k in current_col_anomaly_indices_relative if k < len(base_timestamps_for_y)]
            
            anomaly_details_list = []
            for k_idx, rel_idx in enumerate(current_col_anomaly_indices_relative):
                if rel_idx < len(actual_original_one_step): # 确保索引有效
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
            
            # 打印该污染物的异常检测总结
            print(f"\n目标污染物: {col_name}")
            print(f"  统计信息: 平均绝对误差={mean_error:.4f}, 误差标准差={std_error:.4f}")
            print(f"  异常判定阈值 (误差 > {threshold:.4f})")
            if len(anomaly_timestamps_col) > 0:
                print(f"  检测到 {len(anomaly_timestamps_col)} 个潜在异常点:")
                for detail in anomaly_details_list[:5]: # 打印前5个异常点的详细信息
                    print(f"    - 时间: {detail['timestamp']}, 实际值: {detail['actual_value']:.2f}, "
                          f"预测值: {detail['predicted_value']:.2f}, 误差: {detail['error']:.2f}")
                if len(anomaly_timestamps_col) > 5: 
                    print(f"    ...等 (共 {len(anomaly_timestamps_col)} 个异常点)")
            else: 
                print("  在此阈值下，未检测到异常点。")
        
            # 为该污染物绘制带有异常标记的图表
            # 图表保存路径前缀，确保每个污染物一个文件
            plot_save_prefix_target = os.path.join(self.artifacts_dir, f"anomaly_detection_report_{col_name}") 
            plot_anomalies(
                timestamps_or_indices=base_timestamps_for_y, # X轴：对应y序列的时间戳/索引
                actual_values=actual_original_one_step[:, i], # Y轴：该污染物的实际观测值序列
                anomaly_indices=current_col_anomaly_indices_relative, # 需要标记的异常点在序列中的相对索引
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
        # --- 模型训练流程 ---
        print("\n--- 您已选择：1. 训练新模型 ---")
        custom_artifacts_dir = input(f"请输入模型文件及相关组件的保存目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        train_data_path = input(f"请输入训练数据文件的完整路径 (例如: data/my_training_data.xlsx, 默认为 '{DEFAULT_FILE_PATH}'): ").strip() or DEFAULT_FILE_PATH
        
        # 新增：询问用户是否启用IQR异常值检测
        enable_iqr_input = input(f"是否在训练预处理中启用IQR异常值检测? (y/n, 默认为 {'y' if DEFAULT_ENABLE_IQR_OUTLIER_DETECTION else 'n'}): ").strip().lower()
        if enable_iqr_input == 'y':
            iqr_override = True
        elif enable_iqr_input == 'n':
            iqr_override = False
        else:
            iqr_override = None # 用户输入无效或为空，则使用配置中的默认值（或全局默认值）

        if not os.path.exists(train_data_path):
            print(f"错误: 训练数据文件 '{train_data_path}' 未找到。请检查路径是否正确。")
        else:
            system = AQISystem(artifacts_dir=custom_artifacts_dir) # 使用指定的（或默认的）工件目录
            print(f"开始使用数据 '{train_data_path}' 在目录 '{custom_artifacts_dir}' 中训练新模型...")
            system.train_new_model(train_data_path=train_data_path, enable_iqr_detection_override=iqr_override)
            print("新模型训练流程已完成。")

    elif action == '2':
        # --- 模型预测流程 ---
        print("\n--- 您已选择：2. 使用现有模型进行预测 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        input_data_source = input("请输入用于预测的输入数据文件路径 (CSV 或 Excel格式，应包含足够的历史数据): ").strip()
        
        if not os.path.exists(input_data_source):
            print(f"错误: 预测输入数据文件 '{input_data_source}' 未找到。请检查路径。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir) # 加载指定目录的模型
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
        # --- 异常数据检测流程 ---
        print("\n--- 您已选择：3. 使用现有模型检测异常数据 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        anomaly_data_path = input("请输入用于异常检测的数据文件路径 (CSV 或 Excel格式，应包含实际观测值): ").strip()
        threshold_factor_str = input(f"请输入异常检测的阈值敏感度因子 (例如输入 3.0 表示均值+3倍标准差，默认为 {DEFAULT_ANOMALY_THRESHOLD_FACTOR}): ").strip()
        
        custom_threshold_factor = None # 初始化为None，如果用户输入无效则使用默认
        if threshold_factor_str: # 如果用户有输入
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
                system = AQISystem(artifacts_dir=custom_artifacts_dir) # 加载指定目录的模型
                print(f"开始使用位于 '{custom_artifacts_dir}' 的模型对数据 '{anomaly_data_path}' 进行异常检测...")
                anomaly_report = system.detect_anomalies(data_path_or_df=anomaly_data_path, threshold_factor=custom_threshold_factor)
                # (可以添加对 anomaly_report 的进一步处理或展示)
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