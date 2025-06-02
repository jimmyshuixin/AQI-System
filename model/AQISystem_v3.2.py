# ==============================================================================
# 核心功能:
# 该脚本构建了一个综合系统，利用Transformer神经网络模型对多种空气质量指标（包括AQI和各项污染物浓度）
# 进行多目标预测，并集成了异常数据检测功能。新增了训练日志记录和独立的模型评估模块。
# 控制台输出现在也会自动保存到日志文件。
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
#    - 训练过程中应用学习率动态调整策略（ReduceLROnPlateau）与早停（Early Stopping）机制。
#    - 自动保存详细的训练过程日志到单独文件。
#
# 3. 模型评估 (通过新增的 ModelEvaluator 模块):
#    - 独立的模块，用于评估已训练模型的性能。
#    - 支持在用户提供的测试数据集上进行评估。
#    - 计算多种评估指标：MAE, RMSE, R², MAPE, SMAPE。
#    - 生成实际值与预测值的对比图。
#
# 4. 模型预测 (通过 AQISystem -> ModelPredictor 逻辑集成):
#    - 便捷加载预先训练完成的模型权重及相关的预处理组件。
#    - 对新的外部输入数据执行空气质量预测。
#    - 支持对未来时间段的多目标预测，输出格式与训练数据一致。
#    - 输出的预测值为训练该模型的数据集截止时间的后72小时。
#
# 5. 异常数据检测 (通过 AQISystem 模块):
#    - 利用训练好的模型对输入数据序列中的潜在异常点进行识别。
#    - 提供异常点的可视化展示。
#
# 6. 命令行用户交互界面:
#    - 脚本启动时，通过命令行提示引导用户选择执行模型训练、评估、预测或异常数据检测。
#
# 7. 全局控制台日志记录:
#    - 所有输出到控制台的信息（包括print语句和标准错误）都会被捕获并保存到一个主日志文件中。
# ==============================================================================

# 导入所需的库
import pandas as pd  # 用于数据处理和分析，特别是DataFrame操作
import numpy as np   # 用于数值计算，特别是数组操作
import matplotlib.pyplot as plt # 用于数据可视化，绘制图表
from sklearn.preprocessing import StandardScaler # 用于特征标准化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # 用于模型评估的指标

import torch # PyTorch深度学习框架
import torch.nn as nn # PyTorch神经网络模块
from torch.utils.data import DataLoader, TensorDataset # PyTorch数据加载工具
from torch.optim.lr_scheduler import ReduceLROnPlateau # 学习率调度器，在验证损失不再改善时降低学习率

import math # 数学函数库
import os   # 与操作系统交互，如文件路径操作
import sys  # 提供对Python解释器变量和函数的访问，此处用于重定向标准输出/错误
import copy # 实现深拷贝和浅拷贝
import joblib # 用于保存和加载Python对象，特别是Scikit-learn模型
import json # 用于处理JSON数据格式
import logging # Python内置的日志记录模块
import datetime # 用于处理日期和时间，常用于生成时间戳
import optuna # 超参数优化框架

# --- Matplotlib 中文显示全局设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 设置matplotlib默认字体为黑体，以支持中文显示
    plt.rcParams['axes.unicode_minus'] = False # 解决matplotlib保存图像时负号'-'显示为方块的问题
except Exception as e:
    # 早期打印，此时日志可能尚未完全配置
    print(f"警告: Matplotlib中文字体设置失败: {e}。图表中的中文标签可能无法正常显示。")


# --- 全局核心参数定义 ---
DEFAULT_FILE_PATH = r'data_process\output\南京_AQI_Data.xlsx' # 默认输入数据文件路径
DEFAULT_LOOK_BACK = 72 # 默认回溯窗口大小（即用过去多少个时间步的数据来预测未来）
DEFAULT_HORIZON = 72   # 默认预测范围大小（即预测未来多少个时间步）
DEFAULT_TARGET_COL_NAMES = ['AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] # 默认的目标预测列名列表
DEFAULT_ALL_AVAILABLE_COL_NAMES = [ # 数据文件中所有可能用到的列名
    'date', 'hour', 'AQI', 'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h',
    'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h',
    'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h'
]
DEFAULT_PRIMARY_TARGET_COL_NAME = 'AQI' # 主要目标变量，用于早停、学习率调整等优化过程
DEFAULT_BATCH_SIZE = 32 # 默认的批处理大小，用于模型训练
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择GPU（如果可用）或CPU作为计算设备

# --- 模型及相关文件保存路径与命名规范 ---
MODEL_ARTIFACTS_DIR = r"model\output" # 模型相关文件（模型权重、缩放器、配置等）的保存目录
MODEL_STATE_SAVE_NAME = "best_aqi_transformer_model_adv.pth" # 保存最佳模型状态的文件名
FEATURE_SCALER_SAVE_NAME = "aqi_feature_scaler_adv.pkl" # 保存特征缩放器的文件名
TARGET_SCALERS_SAVE_NAME = "aqi_target_scalers_adv.pkl" # 保存目标变量缩放器的文件名
MODEL_CONFIG_SAVE_NAME = "model_config_adv.json" # 保存模型配置信息的文件名
LOG_DIR_NAME = "logs" # 日志文件保存的子目录名
MAIN_CONSOLE_LOG_FILE_PREFIX = "console_output_" # 主控制台日志文件名的前缀

# --- 训练过程特定参数 ---
DEFAULT_FULL_TRAIN_EPOCHS = 200  # 最终模型训练时的默认轮数
DEFAULT_N_OPTUNA_TRIALS = 150    # Optuna超参数优化时的默认试验次数
DEFAULT_OPTUNA_EPOCHS = 30     # Optuna每次试验训练的默认轮数
DEFAULT_EARLY_STOPPING_PATIENCE = 20 # 早停机制的耐心值（连续多少轮验证损失未改善则停止）
DEFAULT_MIN_DELTA = 0.00001 # 判断验证损失是否有改善的最小变化量
DEFAULT_ANOMALY_THRESHOLD_FACTOR = 3.0 # 异常检测时，判断异常的阈值因子（均值 + factor * 标准差）
DEFAULT_ENABLE_IQR_OUTLIER_DETECTION = False # 是否在训练预处理中启用基于IQR的异常值检测，默认为否

# --- AQI (空气质量指数) 计算相关的国家标准常量 ---
# IAQI (Individual Air Quality Index，个体空气质量指数) 的等级划分标准值
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
# 各类污染物浓度对应的IAQI计算分段点 (单位: µg/m³，CO为mg/m³)
# 这些分段点依据中国《环境空气质量标准》(GB 3095-2012)，并对应特定的污染物浓度平均周期。
POLLUTANT_BREAKPOINTS = {
    'SO2_24h':   [0, 50,  150,  475,  800,  1600, 2100, 2620], # 二氧化硫24小时平均浓度限值 (μg/m³)
    'NO2_24h':   [0, 40,  80,   180,  280,  565,  750,  940],  # 二氧化氮24小时平均浓度限值 (μg/m³)
    'PM10_24h':  [0, 50,  150,  250,  350,  420,  500,  600],  # PM10 24小时平均浓度限值 (μg/m³)
    'CO_24h':    [0, 2,   4,    14,   24,   36,   48,   60],   # 一氧化碳24小时平均浓度限值 (mg/m³)
    'O3_8h_24h': [0, 100, 160,  215,  265,  800, 1000, 1200], # 臭氧8小时滑动平均浓度限值 (μg/m³)
    'O3_1h':     [0, 160, 200,  300,  400,  800, 1000, 1200], # 臭氧1小时平均浓度限值 (μg/m³)
    'PM2.5_24h': [0, 35,  75,   115,  150,  250,  350,  500]   # PM2.5 24小时平均浓度限值 (μg/m³)
}
# 小时浓度近似的污染物浓度限值，当24小时数据不可用时，作为一种近似计算IAQI的参考
POLLUTANT_BREAKPOINTS_HOURLY_APPROX = {
    'SO2':   [0, 150, 500,  650,  800], # 二氧化硫小时浓度近似限值
    'NO2':   [0, 100, 200,  700, 1200], # 二氧化氮小时浓度近似限值
    'PM10':  [0, 50, 150, 250, 350, 420], # PM10小时浓度近似限值
    'CO':    [0, 5,  10,   35,   60],    # 一氧化碳小时浓度近似限值 (mg/m³)
    'O3':    POLLUTANT_BREAKPOINTS['O3_1h'], # 臭氧小时浓度直接使用1小时标准
    'PM2.5': [0, 35, 75, 115, 150, 250] # PM2.5小时浓度近似限值
}

# --- 日志相关类和函数 ---
class StreamToLogger:
    """
    自定义流对象，将写入操作重定向到日志记录器。
    用于捕获标准输出 (stdout) 和标准错误 (stderr) 的内容，并将其作为日志信息记录下来。
    """
    def __init__(self, logger, level):
        """
        初始化StreamToLogger。
        参数:
            logger: logging.Logger 对象，用于记录信息。
            level: 日志级别 (例如 logging.INFO, logging.ERROR)。
        """
        self.logger = logger
        self.level = level
        self.linebuf = '' #行缓冲区

    def write(self, buf):
        """
        实现流的write方法。当有内容写入此流时，此方法被调用。
        参数:
            buf: 字符串，要写入的内容。
        """
        for line in buf.rstrip().splitlines(): # 移除末尾空白并按行分割
            self.logger.log(self.level, line.rstrip()) # 将每行记录到日志

    def flush(self):
        """
        实现流的flush方法。
        日志记录器通常会自动刷新或根据其处理器配置进行刷新，
        但显式调用可以确保所有缓冲的日志都被处理。
        """
        for handler in self.logger.handlers:
            handler.flush()

# 全局标志，确保日志和重定向只设置一次
_log_setup_done = False

def setup_global_logging_and_redirect(log_dir, log_prefix, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    配置全局日志记录系统。
    该函数会设置一个主应用程序日志记录器，使其能够同时输出日志到控制台和指定的日志文件。
    并且，它会将Python的默认标准输出 (sys.stdout) 和标准错误 (sys.stderr) 重定向，
    使得所有通过 print() 语句输出的信息或未捕获的错误信息也能被记录到日志文件中。
    参数:
        log_dir (str): 日志文件存放的目录路径。
        log_prefix (str): 日志文件名的前缀。文件名将是 "前缀_时间戳.log"。
        console_level (int, optional): 控制台输出的日志级别。默认为 logging.INFO。
        file_level (int, optional): 文件记录的日志级别。默认为 logging.DEBUG。
    返回:
        logging.Logger: 配置好的主应用程序日志记录器实例。
    """
    global _log_setup_done # 引用全局标志
    if _log_setup_done: # 如果已经设置过，则直接返回已配置的记录器
        return logging.getLogger("AQIMainApp")

    os.makedirs(log_dir, exist_ok=True) # 创建日志目录（如果不存在）
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # 获取当前时间戳
    log_file_path = os.path.join(log_dir, f"{log_prefix}{timestamp}.log") # 构建完整的日志文件路径

    # 配置根记录器或特定的主应用程序记录器
    # 使用特定名称的记录器 ("AQIMainApp")，而不是根记录器，以避免影响其他库的日志行为
    logger = logging.getLogger("AQIMainApp")
    logger.setLevel(min(console_level, file_level)) # 设置记录器的最低级别，确保能捕获所有需要的日志

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 定义日志格式

    # 控制台处理器 (StreamHandler)
    # 检查是否已经有控制台处理器，避免重复添加
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.__stdout__) # 创建控制台处理器，确保写入原始的stdout
        ch.setLevel(console_level) # 设置控制台处理器的级别
        ch.setFormatter(formatter) # 设置格式
        logger.addHandler(ch) # 将处理器添加到记录器

    # 文件处理器 (FileHandler)
    # 检查是否已经有相同路径的文件处理器，避免重复添加
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        fh = logging.FileHandler(log_file_path, encoding='utf-8') # 创建文件处理器，使用utf-8编码
        fh.setLevel(file_level) # 设置文件处理器的级别
        fh.setFormatter(formatter) # 设置格式
        logger.addHandler(fh) # 将处理器添加到记录器

    # 重定向 stdout 和 stderr 到我们自定义的 StreamToLogger 实例
    sys.stdout = StreamToLogger(logger, logging.INFO) # 所有标准输出将以INFO级别记录
    sys.stderr = StreamToLogger(logger, logging.ERROR) # 所有标准错误将以ERROR级别记录

    _log_setup_done = True # 设置标志为True，表示已完成配置
    logger.info(f"全局日志记录已设置。所有控制台输出将记录到: {log_file_path}")
    return logger


def get_configured_logger(name="AQISystemSubmodule", log_file_path=None, level=logging.INFO):
    """
    获取或配置一个特定的子模块日志记录器。
    如果提供了log_file_path，它会为这个特定的记录器添加一个文件处理器，用于将该模块的日志记录到单独的文件。
    这个子模块记录器默认会继承其父记录器（例如 "AQIMainApp"）的控制台输出行为（如果父记录器已配置）。
    参数:
        name (str, optional): 子模块记录器的名称。默认为 "AQISystemSubmodule"。
        log_file_path (str, optional): 特定子模块日志文件的路径。如果为None，则不添加文件处理器。默认为 None。
        level (int, optional): 该记录器的日志级别。默认为 logging.INFO。
    返回:
        logging.Logger: 配置好的子模块日志记录器实例。
    """
    logger = logging.getLogger(name) # 获取指定名称的记录器实例

    # 确保日志级别被设置，即使没有新的处理器添加（它可能从父级继承处理器）
    # logger.level 为 0 表示未设置级别，此时需要设置
    if not logger.level or logger.level > level:
        logger.setLevel(level)

    # 检查是否已有相同的文件处理器，避免重复添加
    has_similar_file_handler = False
    if log_file_path:
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_file_path:
                has_similar_file_handler = True
                break

    if log_file_path and not has_similar_file_handler: # 如果指定了日志文件且没有重复的处理器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') # 定义日志格式
        fh = logging.FileHandler(log_file_path, encoding='utf-8') # 创建文件处理器
        fh.setLevel(level) # 文件处理器可以有自己的级别
        fh.setFormatter(formatter) # 设置格式
        logger.addHandler(fh) # 添加到记录器
        logger.info(f"模块 '{name}' 的文件日志处理器已添加到: {log_file_path}")

    # logger.propagate 默认为 True，这意味着子记录器的日志消息会向上传播到父记录器。
    # 如果父记录器（如 "AQIMainApp"）配置了控制台处理器，那么子记录器的消息也会通过它显示在控制台。
    # 如果需要独立的控制台输出行为（不传播），可以在这里设置 logger.propagate = False 并添加独立的 StreamHandler。
    # 但通常情况下，我们希望子模块日志通过主应用程序的控制台处理程序显示。

    return logger


# --- 辅助工具函数 ---
initial_logger = logging.getLogger("AQIMainApp") # 早期日志使用主应用记录器，确保在setup_global_logging之后使用

def set_seed(seed_value=42):
    """
    设置随机种子以确保实验的可复现性。
    参数:
        seed_value (int): 要设置的种子值。
    """
    np.random.seed(seed_value) # 设置NumPy的随机种子
    torch.manual_seed(seed_value) # 设置PyTorch CPU的随机种子
    if torch.cuda.is_available(): # 如果CUDA可用
        torch.cuda.manual_seed_all(seed_value) # 设置所有PyTorch GPU的随机种子
        torch.backends.cudnn.deterministic = True # 确保cuDNN使用确定性算法
        torch.backends.cudnn.benchmark = False # 禁用cuDNN的自动基准测试（可能引入不确定性）
    os.environ['PYTHONHASHSEED'] = str(seed_value) # 设置Python内置哈希函数的种子

def calculate_iaqi(Cp, pollutant_key):
    """
    根据污染物浓度Cp和污染物类型，计算单个污染物的空气质量分指数 (IAQI)。
    计算方法基于国家标准 GB 3095-2012 附录A。
    参数:
        Cp (float): 污染物浓度值。
        pollutant_key (str): 污染物键名，如 'PM2.5_24h', 'SO2_24h'等，用于查找对应的浓度限值表。
    返回:
        float: 计算得到的IAQI值，如果无法计算则返回 np.nan。
    """
    if pd.isna(Cp) or Cp < 0: # 如果浓度值无效（NaN或负数）
        return np.nan

    bp_table_to_use = POLLUTANT_BREAKPOINTS # 默认使用24小时平均浓度限值表
    if pollutant_key not in bp_table_to_use: # 如果在24小时表中找不到
        bp_table_to_use = POLLUTANT_BREAKPOINTS_HOURLY_APPROX # 尝试使用小时浓度近似限值表
        if pollutant_key not in bp_table_to_use: # 如果小时表中也找不到
            return np.nan # 无法计算

    bp = bp_table_to_use.get(pollutant_key) # 获取该污染物的浓度限值列表
    if Cp > bp[-1]: # 如果浓度超过最高限值
        return 500 # IAQI直接取500（或标准中定义的最高IAQI级别对应的值）

    for i in range(len(bp) - 1): # 遍历浓度限值区间
        if bp[i] <= Cp < bp[i+1]: # 如果浓度落在某个区间 [BP_Lo, BP_Hi)
            IAQI_Lo, IAQI_Hi = IAQI_LEVELS[i], IAQI_LEVELS[i+1] # 对应的IAQI上下限
            BP_Lo, BP_Hi = bp[i], bp[i+1] # 对应的浓度上下限
            if BP_Hi == BP_Lo: return IAQI_Lo # 避免除以零
            # IAQI计算公式
            return round(((IAQI_Hi - IAQI_Lo) / (BP_Hi - BP_Lo)) * (Cp - BP_Lo) + IAQI_Lo)

    if Cp == bp[0]: # 如果浓度等于最低限值
        return IAQI_LEVELS[0] # IAQI为0

    return np.nan # 其他情况，无法确定区间

def calculate_aqi_from_pollutants(df):
    """
    根据DataFrame中各污染物的浓度数据，计算每条记录的综合空气质量指数 (AQI) 和首要污染物。
    AQI的计算逻辑：
    1. 对每种参与AQI计算的污染物，根据其浓度计算对应的IAQI。
    2. 取所有污染物IAQI中的最大值作为该时刻的AQI。
    3. AQI大于50时，IAQI值等于AQI的污染物即为首要污染物。
    参数:
        df (pd.DataFrame): 包含污染物浓度数据的DataFrame，索引应为时间。
                           期望包含如 'PM2.5_24h', 'SO2_24h' 或其小时近似列。
    返回:
        pd.DataFrame: 更新后的DataFrame，增加了 'AQI_calculated' 和 'Primary_Pollutant_calculated' 两列。
    """
    iaqi_df = pd.DataFrame(index=df.index) # 创建一个空的DataFrame用于存放各污染物的IAQI值

    # 定义参与AQI计算的污染物及其在输入df中可能的列名（优先使用24h平均，其次是小时值）
    pollutants_for_calc = {
        'SO2_24h':   ['SO2_24h', 'SO2'],
        'NO2_24h':   ['NO2_24h', 'NO2'],
        'PM10_24h':  ['PM10_24h', 'PM10'],
        'CO_24h':    ['CO_24h', 'CO'], # 注意CO单位是mg/m3，其他是μg/m3，但calculate_iaqi内部处理
        'O3_8h_24h': ['O3_8h_24h', 'O3_8h', 'O3'], # 臭氧优先8h滑动平均，其次是1h（O3_8h通常指8h滑动平均，O3可能指1h）
        'PM2.5_24h': ['PM2.5_24h', 'PM2.5']
    }

    # 遍历每种污染物，计算IAQI
    for bp_key, df_col_options in pollutants_for_calc.items():
        selected_col_for_iaqi = None
        for df_col in df_col_options: # 尝试从提供的列选项中找到可用的数据列
            if df_col in df.columns and not df[df_col].isnull().all(): # 如果列存在且不全为NaN
                selected_col_for_iaqi = df_col
                break # 找到第一个可用的就使用

        if selected_col_for_iaqi:
            # 使用选定的列数据计算IAQI
            # 如果bp_key在标准24h断点表中，则使用bp_key，否则说明可能用的是小时近似列，此时calculate_iaqi内部会尝试匹配
            iaqi_df[bp_key] = df[selected_col_for_iaqi].apply(
                lambda x: calculate_iaqi(x, bp_key if bp_key in POLLUTANT_BREAKPOINTS else selected_col_for_iaqi)
            )
        else:
            iaqi_df[bp_key] = np.nan # 如果没有可用的数据列，则该污染物的IAQI为NaN

    # AQI是所有IAQI中的最大值
    df['AQI_calculated'] = iaqi_df.max(axis=1, skipna=True)

    # 计算首要污染物
    def get_primary_pollutants(row):
        # 如果AQI_calculated是NaN或者小于等于50（优良），则无首要污染物
        if pd.isna(row['AQI_calculated']) or row['AQI_calculated'] <= 50:
            return '无'
        # 找出IAQI值约等于AQI_calculated的污染物
        primary = [pollutant_bp_key for pollutant_bp_key in iaqi_df.columns
                   if pd.notna(row[pollutant_bp_key]) and round(row[pollutant_bp_key]) == round(row['AQI_calculated'])]
        return ', '.join(primary) if primary else '无' # 可能有多个首要污染物

    # 为了在get_primary_pollutants函数中能访问到iaqi_df和AQI_calculated，创建一个临时DataFrame
    temp_iaqi_df_for_primary = iaqi_df.copy()
    temp_iaqi_df_for_primary['AQI_calculated'] = df['AQI_calculated'] # 将计算出的AQI加入临时表
    df['Primary_Pollutant_calculated'] = temp_iaqi_df_for_primary.apply(get_primary_pollutants, axis=1)

    return df

def create_sequences(data_df, look_back, horizon, target_col_names, feature_cols, is_predict=False, logger=None):
    """
    将时间序列数据转换为适用于监督学习的序列样本 (X, y)。
    X 是输入序列 (look_back 长度)，y 是对应的目标序列 (horizon 长度)。
    参数:
        data_df (pd.DataFrame): 包含特征和目标列的DataFrame，已进行预处理和缩放。
        look_back (int): 输入序列的长度（回溯窗口大小）。
        horizon (int): 目标序列的长度（预测范围）。
        target_col_names (list): 目标列的名称列表。
        feature_cols (list): 用作输入的特征列的名称列表。
        is_predict (bool, optional): 是否为预测模式。在预测模式下，不生成y。默认为False。
        logger (logging.Logger, optional): 日志记录器实例。如果为None，则使用全局初始记录器。
    返回:
        tuple: (np.array, np.array) 或 (np.array, None)
               - X_arr: 输入序列数组，形状为 (样本数, look_back, 特征数)。
               - y_arr: 目标序列数组，形状为 (样本数, horizon, 目标数)。如果 is_predict=True，则为None。
               如果无法创建任何序列（例如数据太短），则返回空的numpy数组。
    """
    current_logger = logger if logger else initial_logger # 获取日志记录器

    X_list, y_list = [], [] # 用于存储生成的序列

    # 检查必要的特征列是否存在
    missing_feature_cols = [col for col in feature_cols if col not in data_df.columns]
    if missing_feature_cols:
        raise ValueError(f"数据DataFrame中缺少必要的特征列: {missing_feature_cols}.")

    # 如果不是预测模式，检查必要的目标列是否存在
    if not is_predict:
        missing_target_cols = [col for col in target_col_names if col not in data_df.columns]
        if missing_target_cols:
            raise ValueError(f"数据DataFrame中缺少必要的目标列: {missing_target_cols}.")

    data_features_np = data_df[feature_cols].values # 将特征数据转换为numpy数组
    if not is_predict:
        data_targets_np = data_df[target_col_names].values # 将目标数据转换为numpy数组

    num_samples = len(data_features_np) # 总数据点数

    # 计算可以生成的序列数量
    if is_predict: # 预测模式下，只需要look_back长度的输入，不需要horizon
        num_possible_sequences = num_samples - look_back + 1
    else: # 训练/评估模式下，需要look_back的输入和horizon的输出
        num_possible_sequences = num_samples - look_back - horizon + 1

    if num_possible_sequences <= 0: # 如果数据长度不足以生成任何序列
        num_features = len(feature_cols)
        num_targets = len(target_col_names) if not is_predict else 0
        empty_x_shape = (0, look_back, num_features) # 定义空的X数组形状
        empty_y_shape = (0, horizon, num_targets) if not is_predict else (0,) # 定义空的y数组形状
        current_logger.warning(f"数据长度 ({num_samples}) 不足以创建任何长度为 (look_back={look_back}, horizon={horizon}) 的序列。")
        return np.empty(empty_x_shape), (np.empty(empty_y_shape) if not is_predict else None)

    # 遍历数据，生成序列
    for i in range(num_possible_sequences):
        X_list.append(data_features_np[i : i + look_back]) # 截取输入序列 X
        if not is_predict:
            y_list.append(data_targets_np[i + look_back : i + look_back + horizon, :]) # 截取目标序列 y

    # 将列表转换为numpy数组
    X_arr = np.array(X_list) if X_list else np.empty((0, look_back, len(feature_cols)))
    if is_predict:
        return X_arr, None # 预测模式下只返回X

    y_arr = np.array(y_list) if y_list else np.empty((0, horizon, len(target_col_names)))

    # 特殊情况处理：如果生成了X但没有生成y（通常发生在数据末尾长度不足以构成完整的horizon）
    if y_arr.size == 0 and X_arr.size > 0 and not is_predict :
        current_logger.warning("已创建输入(X)序列，但由于数据末端长度不足，未能生成对应的输出(y)序列。")
        # 返回空的X和y，以避免后续处理出错
        return np.empty((0, look_back, len(feature_cols))), np.empty((0, horizon, len(target_col_names)))

    return X_arr, y_arr

def plot_training_loss(train_losses, val_losses, save_path, title_prefix=""):
    """
    绘制并保存模型训练过程中的训练损失和验证损失曲线图。
    参数:
        train_losses (list): 每个epoch的训练损失列表。
        val_losses (list): 每个epoch的验证损失列表。
        save_path (str): 图像保存的完整路径（包括文件名和扩展名）。
        title_prefix (str, optional): 图表标题的前缀。默认为空字符串。
    """
    plt.figure(figsize=(10, 6)) # 设置图像大小
    plt.plot(train_losses, label='训练损失 (Train Loss)') # 绘制训练损失曲线
    plt.plot(val_losses, label='验证损失 (Validation Loss)') # 绘制验证损失曲线
    plt.title(f'{title_prefix}模型训练过程中的损失变化') # 设置图表标题
    plt.xlabel('训练轮数 (Epoch)') # 设置X轴标签
    plt.ylabel('损失函数值 (Loss - MSE)') # 设置Y轴标签
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格
    plt.savefig(save_path) # 保存图像
    plt.close() # 关闭图像，释放内存

def plot_predictions_vs_actual(actual, predicted, target_name, save_path_prefix, title_suffix="实际值 vs. 预测值"):
    """
    绘制并保存单个目标变量的实际值与模型预测值的对比曲线图。
    参数:
        actual (np.array): 实际目标值数组，形状可以是 (样本数, horizon) 或已展平。
        predicted (np.array): 模型预测目标值数组，形状与actual一致。
        target_name (str): 当前目标变量的名称（例如 'AQI', 'PM2.5'）。
        save_path_prefix (str): 图像保存路径的前缀。最终文件名会是 "前缀_目标名_predictions.png"。
        title_suffix (str, optional): 图表标题的后缀。默认为 "实际值 vs. 预测值"。
    """
    plt.figure(figsize=(15, 7)) # 设置图像大小
    actual_flat = actual.flatten() # 将实际值展平为一维数组
    predicted_flat = predicted.flatten() # 将预测值展平为一维数组

    plt.plot(actual_flat, label=f'实际值 ({target_name})', alpha=0.7) # 绘制实际值曲线
    plt.plot(predicted_flat, label=f'预测值 ({target_name})', linestyle='--', alpha=0.7) # 绘制预测值曲线

    plt.title(f'{target_name} - {title_suffix}') # 设置图表标题
    plt.xlabel('时间步 (Time Step)') # 设置X轴标签
    plt.ylabel(f'{target_name} 浓度/指数值') # 设置Y轴标签
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格
    plt.savefig(f"{save_path_prefix}_{target_name}_predictions.png") # 保存图像
    plt.close() # 关闭图像

def plot_anomalies(timestamps_or_indices, actual_values, anomaly_indices, target_name, save_path_prefix, title_suffix="异常点"):
    """
    绘制时间序列数据，并在图中标出检测到的异常点。
    参数:
        timestamps_or_indices (array-like): X轴的值，可以是时间戳序列 (pd.DatetimeIndex, pd.RangeIndex) 或简单的数字索引。
        actual_values (np.array): 实际观测值的一维数组。
        anomaly_indices (array-like): 检测到的异常点在 actual_values 中的索引列表或数组。
        target_name (str): 目标变量的名称。
        save_path_prefix (str): 图像保存路径的前缀。最终文件名会是 "前缀_anomalies.png"。
        title_suffix (str, optional): 图表标题的后缀。默认为 "异常点"。
    """
    plt.figure(figsize=(15, 7)) # 设置图像大小
    actual_flat = actual_values.flatten() # 确保实际值是一维的

    # 绘制原始时间序列
    plt.plot(timestamps_or_indices, actual_flat, label=f'实际值 ({target_name})', alpha=0.7)

    # 筛选出有效的异常点索引（确保在数据范围内）
    valid_anomaly_indices = np.array([idx for idx in anomaly_indices if 0 <= idx < len(actual_flat)], dtype=int)

    if len(valid_anomaly_indices) > 0: # 如果存在有效异常点
        # 根据X轴的类型确定异常点的X坐标
        if isinstance(timestamps_or_indices, (pd.DatetimeIndex, pd.RangeIndex)):
            plot_x_values = np.array(timestamps_or_indices) # 如果是Pandas索引，直接使用
        else:
            plot_x_values = np.asarray(timestamps_or_indices) # 其他情况转换为Numpy数组

        anomaly_x = plot_x_values[valid_anomaly_indices] # 获取异常点的X坐标
        anomaly_y = actual_flat[valid_anomaly_indices] # 获取异常点的Y坐标（实际值）
        # 用散点图标记异常点
        plt.scatter(anomaly_x, anomaly_y, color='red', label='检测到的异常点 (Anomaly)', marker='o', s=50, zorder=5)

    plt.title(f'{target_name} - {title_suffix}') # 设置图表标题
    plt.xlabel('时间 / 序列索引') # 设置X轴标签
    plt.ylabel(f'{target_name} 浓度/指数值') # 设置Y轴标签
    plt.legend() # 显示图例
    plt.grid(True) # 显示网格
    plt.tight_layout() # 调整布局以防止标签重叠
    save_full_path = f"{save_path_prefix}_anomalies.png" # 构建完整的保存路径
    plt.savefig(save_full_path) # 保存图像
    plt.close() # 关闭图像

def detect_anomalies_iqr_and_impute(df, column_names, logger, factor=1.5, interpolation_method='time'):
    """
    对DataFrame中指定的列进行基于IQR（四分位距）的异常值检测，并将检测到的异常值替换为NaN，
    然后使用插值方法填充这些NaN以及原有的NaN值。
    参数:
        df (pd.DataFrame): 输入的DataFrame。
        column_names (list): 需要进行异常值处理的列名列表。
        logger (logging.Logger): 日志记录器实例。
        factor (float, optional): IQR的倍数，用于定义异常值的边界。默认为1.5。
                                  下界 = Q1 - factor * IQR, 上界 = Q3 + factor * IQR。
        interpolation_method (str, optional): Pandas插值方法。如果索引是DatetimeIndex，默认为'time'；
                                             否则默认为'linear'。
    返回:
        pd.DataFrame: 处理了异常值并填充了NaN的DataFrame副本。
    """
    df_cleaned = df.copy() # 创建副本以避免修改原始DataFrame
    logger.info("开始对指定列进行基于IQR的异常值检测和插值填充...")

    for col_name in column_names: # 遍历需要处理的每一列
        if col_name in df_cleaned.columns: # 确保列存在
            if not pd.api.types.is_numeric_dtype(df_cleaned[col_name]): # 检查是否为数值类型
                logger.warning(f"列 '{col_name}' 非数值类型，跳过其异常值处理流程。")
                continue

            Q1 = df_cleaned[col_name].quantile(0.25) # 计算第一四分位数
            Q3 = df_cleaned[col_name].quantile(0.75) # 计算第三四分位数
            IQR = Q3 - Q1 # 计算四分位距

            if pd.notna(IQR) and IQR > 1e-6: # 仅当IQR有效且大于一个很小的值时进行处理（避免IQR为0的情况）
                lower_bound = Q1 - factor * IQR # 计算异常值下界
                upper_bound = Q3 + factor * IQR # 计算异常值上界
                # 识别异常值
                outlier_mask = (df_cleaned[col_name] < lower_bound) | (df_cleaned[col_name] > upper_bound)
                num_outliers = outlier_mask.sum() # 统计异常值数量

                if num_outliers > 0:
                    logger.info(f"列 '{col_name}': 检测到 {num_outliers} 个基于IQR的统计学异常值。将其标记为NaN以便后续插值。")
                    df_cleaned.loc[outlier_mask, col_name] = np.nan # 将异常值替换为NaN
            else:
                logger.info(f"列 '{col_name}': IQR值为0或无效 ({IQR})，跳过基于IQR的异常值标记步骤。")

            # 对该列进行插值填充（包括之前标记的异常值NaN和原有的NaN）
            if isinstance(df_cleaned.index, pd.DatetimeIndex): # 如果索引是时间类型
                try:
                    # 优先使用指定的时间插值方法
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method=interpolation_method, limit_direction='both')
                except Exception as e: # 如果时间插值失败
                    logger.warning(f"列 '{col_name}' 时间插值失败: {e}。尝试使用线性插值作为备选。")
                    df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')
            else: # 如果索引不是时间类型，使用线性插值
                df_cleaned[col_name] = df_cleaned[col_name].interpolate(method='linear', limit_direction='both')

            # 如果插值后仍有NaN（通常是序列开头或结尾无法插值的情况），使用中位数或0填充
            if df_cleaned[col_name].isna().sum() > 0 :
                median_val = df_cleaned[col_name].median() # 计算中位数
                fill_value = median_val if pd.notna(median_val) else 0 # 如果中位数有效则用中位数，否则用0
                df_cleaned[col_name] = df_cleaned[col_name].fillna(fill_value)
                logger.info(f"列 '{col_name}': 剩余的NaN值已使用中位数({median_val:.2f})或0 (若中位数无效，则为{fill_value:.2f})进行填充。")
        else:
            logger.warning(f"列 '{col_name}' 在DataFrame中未找到，跳过其异常值处理。")

    logger.info("基于IQR的异常值检测和插值填充流程完成。")
    return df_cleaned

class TimeSeriesDataset(TensorDataset):
    """
    自定义的PyTorch数据集类，用于时间序列数据。
    继承自TensorDataset，它将输入的NumPy数组X和y转换为PyTorch张量。
    """
    def __init__(self, X, y):
        """
        初始化TimeSeriesDataset。
        参数:
            X (np.array): 输入序列数据，形状 (样本数, look_back, 特征数)。
            y (np.array): 目标序列数据，形状 (样本数, horizon, 目标数)。
        """
        X_tensor = torch.from_numpy(X).float() # 将X转换为float类型的PyTorch张量
        y_tensor = torch.from_numpy(y).float() # 将y转换为float类型的PyTorch张量
        super(TimeSeriesDataset, self).__init__(X_tensor, y_tensor) # 调用父类的构造函数

class AQITransformer(nn.Module):
    """
    基于Transformer Encoder的空气质量指数（AQI）预测模型。
    该模型接收一个历史时间序列作为输入，并预测未来多个时间步的多个目标变量。
    """
    def __init__(self, num_features, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, horizon, num_target_features, norm_first=True):
        """
        初始化AQITransformer模型。
        参数:
            num_features (int): 输入特征的数量（每个时间步的特征维度）。
            d_model (int): Transformer模型内部的特征维度 (embedding dimension)。
            nhead (int): Transformer编码器层中多头注意力机制的头数。d_model必须能被nhead整除。
            num_encoder_layers (int): Transformer编码器中的层数。
            dim_feedforward (int): Transformer编码器层中前馈神经网络的隐藏层维度。
            dropout (float): Transformer编码器层中的dropout概率。
            horizon (int): 预测的时间范围长度。
            num_target_features (int): 需要预测的目标变量的数量。
            norm_first (bool, optional): 是否在Transformer编码器层中先进行层归一化再进行其他操作。
                                         PyTorch 1.9+ 推荐设置为True以获得更好的性能和稳定性。默认为True。
        """
        super(AQITransformer, self).__init__()
        self.d_model = d_model # 模型内部维度
        self.horizon = horizon # 预测范围
        self.num_target_features = num_target_features # 目标特征数量

        # 输入嵌入层：将原始特征维度的输入线性变换到d_model维度
        self.input_embedding = nn.Linear(num_features, d_model)

        # 定义Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,          # 模型维度
            nhead=nhead,              # 多头注意力头数
            dim_feedforward=dim_feedforward, # 前馈网络隐藏层维度
            dropout=dropout,          # Dropout概率
            activation='gelu',        # 激活函数，GELU通常表现较好
            batch_first=True,         # 输入和输出张量的形状为 (batch, seq, feature)
            norm_first=norm_first     # 是否先执行LayerNorm
        )
        # 堆叠多个编码器层形成完整的Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 输出层：将Transformer编码器的输出（通常取最后一个时间步的表示）映射到预测结果
        # 输出维度为 horizon * num_target_features，之后会reshape为 (batch, horizon, num_target_features)
        self.output_layer = nn.Linear(d_model, horizon * num_target_features)

    def forward(self, src):
        """
        模型的前向传播。
        参数:
            src (torch.Tensor): 输入的时间序列数据，形状为 (batch_size, look_back, num_features)。
        返回:
            torch.Tensor: 模型的预测输出，形状为 (batch_size, horizon, num_target_features)。
        """
        # 1. 输入嵌入和缩放
        src_embedded = self.input_embedding(src) * math.sqrt(self.d_model) # 乘以sqrt(d_model)是Transformer中的常见做法

        # 2. 位置编码 (Positional Encoding)
        seq_len = src_embedded.size(1) # 获取输入序列的长度 (look_back)
        pe = torch.zeros(seq_len, self.d_model).to(src_embedded.device) # 初始化位置编码张量
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(src_embedded.device) # 位置索引 (0, 1, ..., seq_len-1)
        # 计算div_term，用于sin和cos函数中的频率
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(src_embedded.device)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数维度使用sin
        if self.d_model % 2 != 0: # 如果d_model是奇数，最后一个cos的维度会少一个
            pe[:, 1::2] = torch.cos(position * div_term)[:,:pe[:,1::2].size(1)] # 奇数维度使用cos，并确保维度匹配
        else:
            pe[:, 1::2] = torch.cos(position * div_term) # 奇数维度使用cos

        src_pos_encoded = src_embedded + pe.unsqueeze(0) # 将位置编码加到嵌入后的输入上 (广播机制)
        src_pos_encoded = nn.Dropout(0.1)(src_pos_encoded) # 应用dropout

        # 3. Transformer编码器处理
        encoder_output = self.transformer_encoder(src_pos_encoded) # (batch_size, seq_len, d_model)

        # 4. 获取用于预测的表示
        # 通常使用编码器输出序列的最后一个时间步的表示作为整个输入序列的聚合表示
        prediction_input = encoder_output[:, -1, :] # (batch_size, d_model)

        # 5. 输出层进行预测
        output_flat = self.output_layer(prediction_input) # (batch_size, horizon * num_target_features)

        # 6. Reshape输出到期望的形状
        output = output_flat.view(output_flat.size(0), self.horizon, self.num_target_features) # (batch_size, horizon, num_target_features)

        return output

class ModelTrainer:
    """
    模型训练器类，负责整个模型训练的流程，包括数据加载、预处理、
    超参数优化（使用Optuna）、模型训练、评估和保存。
    """
    def __init__(self, config, logger):
        """
        初始化ModelTrainer。
        参数:
            config (dict): 包含训练所需所有配置参数的字典。
                           例如：文件路径、look_back、horizon、批大小、轮数、优化器参数等。
            logger (logging.Logger): 日志记录器实例。
        """
        self.config = config # 存储配置信息
        self.logger = logger # 存储日志记录器
        self.all_feature_columns_for_sequence = [] # 存储最终用于创建序列的特征列名列表
        self.feature_scaler = None # 特征缩放器 (StandardScaler)
        self.target_scalers = {}   # 目标变量缩放器字典，键为目标列名，值为对应的StandardScaler

    def _load_and_preprocess_data_core(self, file_path, fit_scalers=True):
        """
        核心的数据加载和预处理函数。
        执行步骤包括：
        1. 从CSV或Excel文件加载数据。
        2. 解析时间戳并设置DataFrame索引。
        3. 将所有数据列转换为数值类型，无法转换的强制为NaN。
        4. 对特征列进行前向填充(ffill)、后向填充(bfill)和0填充处理NaN。
        5. (可选，训练时) 对目标列，如果启用IQR检测，则进行IQR异常值检测与插值平滑。
        6. (可选，训练时) 删除目标列中仍存在的NaN值的行。
        7. 根据污染物浓度计算AQI ('AQI_calculated') 并作为特征。
        8. 创建周期性时间特征 (如小时、星期的sin/cos编码)。
        9. 为目标变量和关键的24h/8h平均浓度列创建滞后特征。
        10. (可选，训练时) 拟合特征缩放器 (StandardScaler) 和各目标变量的缩放器，并保存它们。
        11. 应用缩放器转换数据。
        参数:
            file_path (str): 数据文件的路径。
            fit_scalers (bool, optional): 是否拟合和保存缩放器。在训练时应为True，
                                          在仅预测或评估已训练模型时，应加载已保存的缩放器而不重新拟合。
                                          此函数内部主要处理fit_scalers=True的情况，加载逻辑在预测器/评估器中。
        返回:
            pd.DataFrame: 经过完整预处理和缩放后的DataFrame。
        异常:
            FileNotFoundError: 如果文件路径无效。
            ValueError: 如果数据处理后为空，或缺少必要的列等。
        """
        self.logger.info(f"开始从文件加载并执行核心数据预处理流程: {file_path}...")
        # 1. 加载数据 (尝试CSV，然后Excel)
        try:
            df = pd.read_csv(file_path)
        except Exception: # 如果CSV读取失败
            try:
                df = pd.read_excel(file_path)
            except Exception as e_excel:
                self.logger.error(f"错误: 读取CSV和Excel文件均失败。路径: '{file_path}'. 详细错误: {e_excel}")
                raise # 重新抛出异常

        # 2. 时间戳解析和索引设置
        if 'date' in df.columns and 'hour' in df.columns: # 检查是否存在'date'和'hour'列
            try:
                # 合并date和hour列创建时间戳字符串，并转换为datetime对象
                df['timestamp'] = pd.to_datetime(df['date'].astype(str) + df['hour'].astype(int).astype(str).str.zfill(2), format='%Y%m%d%H')
                df = df.set_index('timestamp').drop(columns=['date', 'hour'], errors='ignore') # 设置为索引并删除原列
            except Exception as e_dt:
                self.logger.warning(f"警告: 从'date'和'hour'列创建时间戳索引失败: {e_dt}。尝试其他方法...")
        elif 'Time' in df.columns: # 检查是否存在'Time'列
             try:
                df['timestamp'] = pd.to_datetime(df['Time']) # 直接转换'Time'列
                df = df.set_index('timestamp').drop(columns=['Time'], errors='ignore')
             except Exception as e_t:
                self.logger.warning(f"警告: 从'Time'列创建时间戳索引失败: {e_t}。尝试其他方法...")
        else: # 如果没有明确的时间列，尝试将第一列作为时间索引
            try:
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]) # 转换第一列为datetime
                df = df.set_index(df.columns[0]) # 将第一列设为索引
            except Exception as e_first_col:
                raise ValueError(f"错误: 无法自动从数据中解析并设置时间索引。失败详情: {e_first_col}")

        # 3. 数值类型转换
        for col in df.columns: # 遍历所有列
            df[col] = pd.to_numeric(df[col], errors='coerce') # 转换为数值，无法转换的设为NaN

        # 4. 特征列NaN值填充
        # 找出非目标列作为候选特征列
        feature_candidate_cols = [col for col in df.columns if col not in self.config['target_col_names']]
        for col in feature_candidate_cols:
            if df[col].isnull().any(): # 如果该特征列有NaN
                # 使用前向填充，然后后向填充，最后用0填充剩余的NaN
                df[col] = df[col].ffill().bfill().fillna(0)

        # 5. & 6. 目标列NaN处理 和 IQR异常值检测 (仅在fit_scalers为True，即训练时)
        if fit_scalers: # 通常在训练时为True
            original_len = len(df)
            # 首先删除目标列中包含任何NaN的行，因为目标值必须有效
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
            if len(df) < original_len:
                self.logger.info(f"信息: 由于目标列存在NaN值，已删除 {original_len - len(df)} 行数据。")

        if df.empty: # 如果处理后数据为空
            raise ValueError("错误: 数据在初步NaN处理后变为空。")

        # (仅训练时) IQR异常值检测和插值
        if fit_scalers and self.config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION):
            self.logger.info("信息: IQR异常值检测功能已启用。开始对目标列进行处理...")
            df = detect_anomalies_iqr_and_impute(df, self.config['target_col_names'], self.logger)
            # IQR处理后可能引入新的NaN（如果插值不完全），再次删除目标列的NaN
            original_len_after_iqr = len(df)
            df = df.dropna(axis=0, how='any', subset=self.config['target_col_names'])
            if len(df) < original_len_after_iqr:
                 self.logger.info(f"信息: 在IQR异常处理和插值后，由于目标列仍存在NaN，又删除了 {original_len_after_iqr - len(df)} 行。")
        elif fit_scalers: # 如果是训练模式但禁用了IQR
            self.logger.info("信息: IQR异常值检测功能已禁用。跳过对目标列的IQR处理步骤。")

        if df.empty: # 如果在IQR处理后数据变为空
            raise ValueError("错误: 数据在IQR异常值处理流程或初步NaN处理后变为空。")

        # 7. 计算AQI作为特征
        df = calculate_aqi_from_pollutants(df.copy()) # 创建副本以避免SettingWithCopyWarning
        if 'AQI_calculated' in df.columns and df['AQI_calculated'].isnull().any():
            df['AQI_calculated'] = df['AQI_calculated'].fillna(0) # 对计算出的AQI中的NaN用0填充

        # 8. 创建周期性时间特征 (sin/cos编码)
        new_cyclical_features = pd.DataFrame(index=df.index) # 创建空的DataFrame存放新特征
        if isinstance(df.index, pd.DatetimeIndex): # 确保索引是时间类型
            idx = df.index
            new_cyclical_features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['month_sin'] = np.sin(2 * np.pi * idx.month / 12.0)
            new_cyclical_features['month_cos'] = np.cos(2 * np.pi * idx.month / 12.0)
            df = pd.concat([df, new_cyclical_features], axis=1) # 合并到主DataFrame
        else:
            self.logger.warning("警告: DataFrame索引不是DatetimeIndex，无法创建周期性时间特征。")

        # 9. 创建滞后特征
        # 滞后特征的数量，取look_back的1/4和1之间的较大值，确保至少有1阶滞后
        num_lags_to_create = max(1, self.config['look_back'] // 4)
        lag_features_to_concat = [] # 存储生成的滞后特征Series
        lag_cols_created_names = [] # 存储滞后特征的列名，用于后续dropna
        # 需要创建滞后特征的列：所有目标列 + DataFrame中所有以'_24h'或'_8h'结尾的非目标列
        cols_for_lags = self.config['target_col_names'] + \
                        [col for col in df.columns if (col.endswith('_24h') or col.endswith('_8h')) and col not in self.config['target_col_names']]
        cols_for_lags = sorted(list(set(cols_for_lags))) #去重并排序

        for col_to_lag in cols_for_lags: # 遍历每一列
            if col_to_lag in df.columns: # 确保列存在
                for lag in range(1, num_lags_to_create + 1): # 创建1到num_lags_to_create阶滞后
                    lag_col_name = f"{col_to_lag}_lag_{lag}" # 滞后特征列名
                    lag_features_to_concat.append(df[col_to_lag].shift(lag).rename(lag_col_name)) # 创建并重命名
                    lag_cols_created_names.append(lag_col_name)

        if lag_features_to_concat: # 如果生成了滞后特征
            df = pd.concat([df] + lag_features_to_concat, axis=1) # 合并到主DataFrame
            original_len_before_lag_dropna = len(df)
            # 创建滞后特征会引入NaN（序列开头部分），需要删除这些行
            df = df.dropna(subset=lag_cols_created_names, how='any')
            if len(df) < original_len_before_lag_dropna:
                self.logger.info(f"信息: 由于创建滞后特征引入NaN，已删除 {original_len_before_lag_dropna - len(df)} 行数据。")

        if df.empty: # 如果在滞后特征处理后数据变为空
            raise ValueError("错误: 数据在创建滞后特征并处理相关NaN后变为空。")

        # 确定最终的数值特征列列表 (self.all_feature_columns_for_sequence)
        temp_feature_cols = []
        for col in df.columns:
            if col in self.config['target_col_names']: # 目标列不是特征
                continue
            if col == 'Primary_Pollutant_calculated': # 'Primary_Pollutant_calculated'是字符串，不是数值特征
                self.logger.info(f"列 '{col}' 被显式排除在数值特征之外 (因其为分类字符串)。")
                continue

            original_dtype = df[col].dtype # 记录原始数据类型
            # 再次尝试转换为数值，以处理可能在特征工程中引入的非数值类型
            series_numeric = pd.to_numeric(df[col], errors='coerce')

            # 特殊处理：如果一列全是NaN，且原始类型不是数字也不是object（可能是空的datetime等），则跳过
            if series_numeric.isnull().all() and not pd.api.types.is_numeric_dtype(original_dtype) and str(original_dtype) != 'object':
                 pass # 允许这类列被跳过，不作为特征

            if series_numeric.isnull().any(): # 如果转换后仍有NaN
                self.logger.info(f"特征候补列 '{col}' (原始类型: {original_dtype}) 包含NaN值（转换后），将用0填充。")
                df[col] = series_numeric.fillna(0) # 用0填充
            else:
                df[col] = series_numeric # 更新列数据为转换后的数值型数据

            if pd.api.types.is_numeric_dtype(df[col]): # 再次确认是数值类型
                temp_feature_cols.append(col)
            else:
                self.logger.warning(f"列 '{col}' (类型: {df[col].dtype}) 在最终检查后仍非数值类型，将从特征列表中排除。")

        self.all_feature_columns_for_sequence = sorted(list(set(temp_feature_cols))) # 去重并排序

        # 确保 'AQI_calculated' (如果存在且是数值型) 被包含在特征中 (如果它不是目标列的话)
        if 'AQI_calculated' in df.columns and \
           pd.api.types.is_numeric_dtype(df['AQI_calculated']) and \
           'AQI_calculated' not in self.config['target_col_names'] and \
           'AQI_calculated' not in self.all_feature_columns_for_sequence:
                 self.all_feature_columns_for_sequence.append('AQI_calculated')
                 self.all_feature_columns_for_sequence = sorted(list(set(self.all_feature_columns_for_sequence)))

        if not self.all_feature_columns_for_sequence: # 如果没有有效的特征列
            raise ValueError("错误: 预处理后没有有效的数值输入特征列可供模型使用。")
        self.logger.info(f"最终用于序列创建的特征列: {self.all_feature_columns_for_sequence}")

        # 检查目标列是否都存在于DataFrame中，为后续缩放做准备
        missing_targets_for_scaling = [tc for tc in self.config['target_col_names'] if tc not in df.columns]
        if missing_targets_for_scaling:
            raise ValueError(f"错误: 一个或多个目标列在准备进行数据缩放前未在DataFrame中找到: {missing_targets_for_scaling}.")

        # 10. & 11. 拟合缩放器并转换数据 (仅在fit_scalers为True，即训练时)
        if fit_scalers:
            self.logger.info("开始拟合和保存数据缩放器...")
            self.feature_scaler = StandardScaler() # 初始化特征缩放器
            # 获取当前DataFrame中实际存在的特征列，与self.all_feature_columns_for_sequence取交集
            current_features_to_scale = [f_col for f_col in self.all_feature_columns_for_sequence if f_col in df.columns]

            if len(current_features_to_scale) != len(self.all_feature_columns_for_sequence):
                missing_cols = set(self.all_feature_columns_for_sequence) - set(current_features_to_scale)
                self.logger.error(f"错误: 部分选定的特征列在准备缩放时未在DataFrame中找到: {missing_cols}")
                # 这里应该抛出异常或采取其他错误处理，因为特征列表不一致会导致后续问题

            # 再次确保所有待缩放的特征列都是数值类型
            for f_col in current_features_to_scale:
                if not pd.api.types.is_numeric_dtype(df[f_col]):
                    self.logger.error(f"严重错误: 特征列 '{f_col}' (类型: {df[f_col].dtype}) 在缩放前仍非数值。")
                    try:
                        df[f_col] = df[f_col].astype(float) # 尝试强制转换为float
                        if not pd.api.types.is_numeric_dtype(df[f_col]): # 再次检查
                             raise ValueError(f"特征列 '{f_col}' 即使在astype(float)后也无法转换为数值。")
                    except ValueError as e_final_convert:
                        raise ValueError(f"错误: 特征列 '{f_col}' 最终无法转换为数值类型以进行缩放。原错误: {e_final_convert}")

            if not current_features_to_scale: # 如果没有特征列可供缩放
                raise ValueError("错误: 没有有效的特征列可用于拟合特征缩放器。")

            # 拟合特征缩放器并转换特征数据
            df[current_features_to_scale] = self.feature_scaler.fit_transform(df[current_features_to_scale])
            # 保存特征缩放器
            joblib.dump(self.feature_scaler, os.path.join(self.config['model_artifacts_dir'], FEATURE_SCALER_SAVE_NAME))

            # 为每个目标变量拟合独立的缩放器并转换数据
            self.target_scalers = {}
            for col_name in self.config['target_col_names']:
                # 确保目标列是数值类型，如果不是，尝试转换并填充NaN为0
                if not pd.api.types.is_numeric_dtype(df[col_name]):
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0)
                scaler = StandardScaler() # 为每个目标列创建一个新的缩放器
                df[[col_name]] = scaler.fit_transform(df[[col_name]]) # 拟合和转换（注意要传入2D数组）
                self.target_scalers[col_name] = scaler # 保存缩放器
            # 保存所有目标变量的缩放器字典
            joblib.dump(self.target_scalers, os.path.join(self.config['model_artifacts_dir'], TARGET_SCALERS_SAVE_NAME))
            self.logger.info("特征缩放器和各目标变量的缩放器均已成功拟合和保存。")

        return df # 返回预处理和（如果fit_scalers=True）缩放后的DataFrame

    def _train_model_core(self, model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, trial=None):
        """
        核心模型训练循环。
        参数:
            model (nn.Module): 待训练的PyTorch模型。
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            criterion (nn.Module): 损失函数 (例如 nn.MSELoss)。
            optimizer (torch.optim.Optimizer): 优化器 (例如 AdamW)。
            scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器 (例如 ReduceLROnPlateau)。
            epochs (int): 训练的总轮数。
            trial (optuna.trial.Trial, optional): Optuna的试验对象，用于超参数优化。如果不是在Optuna流程中，则为None。
        返回:
            tuple:
                - model (nn.Module): 训练完成（或早停时最佳状态）的模型。
                - train_losses_epoch (list): 每轮的平均训练损失列表。
                - val_losses_epoch (list): 每轮的平均验证损失列表。
                - best_val_loss (float): 训练过程中达到的最佳验证损失（基于主要目标）。
        """
        best_val_loss = float('inf') # 初始化最佳验证损失为正无穷大
        epochs_no_improve = 0        # 记录验证损失连续未改善的轮数，用于早停
        best_model_state = None      # 存储最佳模型参数状态
        train_losses_epoch = []      # 记录每轮的训练损失
        val_losses_epoch = []        # 记录每轮的验证损失

        # 获取主要目标变量在目标列表中的索引，用于早停和学习率调整
        primary_target_idx = self.config['target_col_names'].index(self.config['primary_target_col_name'])

        for epoch in range(epochs): # 遍历每个训练轮次
            model.train() # 设置模型为训练模式
            running_train_loss = 0.0 # 当前轮次的累计训练损失

            # --- 训练阶段 ---
            for X_batch, y_batch in train_loader: # 遍历训练数据批次
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE) # 数据移至计算设备
                optimizer.zero_grad()  # 清空之前的梯度
                outputs = model(X_batch) # 模型前向传播，得到预测输出
                loss = criterion(outputs, y_batch) # 计算损失
                loss.backward() # 反向传播，计算梯度
                optimizer.step() # 更新模型参数
                running_train_loss += loss.item() * X_batch.size(0) # 累加损失 (乘以批大小以得到总损失)

            # 计算当前轮次的平均训练损失
            epoch_train_loss = running_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0
            train_losses_epoch.append(epoch_train_loss)

            # --- 验证阶段 ---
            model.eval() # 设置模型为评估模式
            running_val_loss = 0.0 # 当前轮次的累计验证损失 (所有目标)
            running_primary_target_val_loss = 0.0 # 当前轮次的累计主要目标验证损失

            if len(val_loader.dataset) > 0: # 确保验证集不为空
                with torch.no_grad(): # 在评估模式下，不计算梯度
                    for X_batch, y_batch in val_loader: # 遍历验证数据批次
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        outputs = model(X_batch) # 模型前向传播
                        loss = criterion(outputs, y_batch) # 计算整体损失
                        running_val_loss += loss.item() * X_batch.size(0)

                        # 计算主要目标变量的损失
                        primary_target_loss = criterion(outputs[:, :, primary_target_idx], y_batch[:, :, primary_target_idx])
                        running_primary_target_val_loss += primary_target_loss.item() * X_batch.size(0)

                # 计算当前轮次的平均验证损失
                epoch_val_loss = running_val_loss / len(val_loader.dataset)
                epoch_primary_target_val_loss = running_primary_target_val_loss / len(val_loader.dataset)
            else: # 如果验证集为空
                epoch_val_loss = float('inf') # 设置为无穷大，避免影响判断
                epoch_primary_target_val_loss = float('inf')

            val_losses_epoch.append(epoch_val_loss) # 记录验证损失

            current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
            self.logger.info(f"轮次 [{epoch+1}/{epochs}], 学习率: {current_lr:.7f}, "
                  f"训练损失: {epoch_train_loss:.6f}, "
                  f"验证损失 (综合): {epoch_val_loss:.6f}, "
                  f"验证损失 ({self.config['primary_target_col_name']}): {epoch_primary_target_val_loss:.6f}")

            # 学习率调度 (基于主要目标的验证损失)
            if scheduler:
                scheduler.step(epoch_primary_target_val_loss)

            # Optuna剪枝 (如果正在进行Optuna优化)
            if trial:
                trial.report(epoch_primary_target_val_loss, epoch) # 向Optuna报告当前试验的性能
                if trial.should_prune(): # 检查是否应该剪枝
                    self.logger.info("Optuna试验被剪枝 (Optuna trial pruned).")
                    raise optuna.exceptions.TrialPruned() # 抛出剪枝异常

            # 早停逻辑 (基于主要目标的验证损失)
            # 如果当前主要目标验证损失比历史最佳值改善超过min_delta
            if epoch_primary_target_val_loss < best_val_loss - self.config['min_delta']:
                best_val_loss = epoch_primary_target_val_loss # 更新最佳验证损失
                epochs_no_improve = 0 # 重置未改善轮数计数器
                best_model_state = copy.deepcopy(model.state_dict()) # 保存当前模型状态为最佳状态
                if trial is None: # 如果不是Optuna试验（即最终模型训练），则保存模型
                    torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME))
                    self.logger.info(f"验证损失 ({self.config['primary_target_col_name']}) 获得改善。模型状态已于轮次 {epoch+1} 保存。")
            else: # 如果没有改善
                epochs_no_improve += 1 # 未改善轮数加1

            # 检查是否达到早停条件
            if epochs_no_improve >= self.config['early_stopping_patience'] and len(val_loader.dataset) > 0 :
                self.logger.info(f"早停机制在轮次 {epoch+1} 被触发 (基于 {self.config['primary_target_col_name']} 的验证损失连续 {self.config['early_stopping_patience']} 轮未改善)。")
                if best_model_state: # 如果有保存过最佳模型
                    model.load_state_dict(best_model_state) # 加载最佳模型状态
                break # 结束训练循环

        # 训练结束后 (正常结束或早停)
        if best_model_state and trial is None: # 如果是最终模型训练且找到了最佳状态
             model.load_state_dict(best_model_state) # 确保加载的是最佳模型
             torch.save(best_model_state, os.path.join(self.config['model_artifacts_dir'], MODEL_STATE_SAVE_NAME)) # 再次保存以防万一
             self.logger.info(f"训练结束。最终最佳模型状态已保存至 {MODEL_STATE_SAVE_NAME}")

        return model, train_losses_epoch, val_losses_epoch, best_val_loss

    def _objective_optuna(self, trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features):
        """
        Optuna的目标函数，用于超参数搜索。
        对于每一组由Optuna建议的超参数，此函数会构建、训练并评估一个模型，
        然后返回一个评估指标（通常是验证集上的损失）给Optuna，以便其进行优化。
        参数:
            trial (optuna.trial.Trial): Optuna的试验对象，用于建议和记录超参数。
            X_train_np, y_train_np (np.array): 训练集的输入和目标序列。
            X_val_np, y_val_np (np.array): 验证集的输入和目标序列。
            num_input_features (int): 输入特征的数量。
        返回:
            float: 该试验在验证集上的最佳损失值（基于主要目标）。
        """
        # --- 1. Optuna建议超参数 ---
        # 学习率 (对数均匀分布)
        lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
        # Transformer模型维度 (从给定列表中选择)
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 512])
        # 多头注意力头数 (确保d_model能被头数整除)
        possible_num_heads = [h for h in [2, 4, 8, 16] if d_model % h == 0 and d_model >= h]
        if not possible_num_heads: # 如果没有有效的头数选项，剪枝此试验
            raise optuna.exceptions.TrialPruned("对于当前d_model，没有有效的注意力头数选项。")
        num_heads = trial.suggest_categorical('num_heads', possible_num_heads)
        # Transformer编码器层数 (整数范围)
        num_encoder_layers = trial.suggest_int('num_encoder_layers', 2, 8)
        # 前馈网络隐藏层维度相对于d_model的倍数 (整数范围)
        dim_feedforward_factor = trial.suggest_int('dim_feedforward_factor', 2, 6)
        dim_feedforward = d_model * dim_feedforward_factor # 计算实际的前馈网络维度
        # Dropout率 (浮点数范围)
        dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.4)
        # 是否先执行LayerNorm (布尔选择)
        norm_first = trial.suggest_categorical('norm_first', [True, False])
        # AdamW优化器的权重衰减 (对数均匀分布)
        weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-3, log=True)
        # ReduceLROnPlateau学习率调度器的衰减因子 (浮点数范围)
        scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.7)
        # ReduceLROnPlateau学习率调度器的耐心值 (整数范围)
        scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)

        # --- 2. 构建模型、优化器、调度器和损失函数 ---
        model = AQITransformer(
            num_features=num_input_features, d_model=d_model, nhead=num_heads,
            num_encoder_layers=num_encoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, norm_first=norm_first,
            horizon=self.config['horizon'], num_target_features=len(self.config['target_col_names'])
        ).to(DEVICE) # 模型移至计算设备

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # AdamW优化器
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience) # 学习率调度器
        criterion = nn.MSELoss() # 均方误差损失

        # --- 3. 创建数据加载器 ---
        # num_workers=0 和 pin_memory=True 是在Windows上使用DataLoader时常见的设置，有助于避免一些问题
        train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        self.logger.info(f"\nOptuna试验 {trial.number}: lr={lr:.6f}, d_model={d_model}, heads={num_heads}, layers={num_encoder_layers}, "
              f"ff_factor={dim_feedforward_factor}, dropout={dropout_rate:.3f}, norm_first={norm_first}, "
              f"wd={weight_decay:.7f}, sch_factor={scheduler_factor:.2f}, sch_patience={scheduler_patience}")

        # --- 4. 调用核心训练循环 ---
        # 使用 self.config['optuna_epochs'] 作为当前试验的训练轮数
        _, _, _, best_val_loss_trial = self._train_model_core(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=self.config['optuna_epochs'], trial=trial # 传入trial对象用于剪枝
        )
        return best_val_loss_trial # 返回此试验的最佳验证损失

    def run_training_pipeline(self):
        """
        执行完整的模型训练流程，包括：
        1. 加载和预处理数据。
        2. 将数据划分为训练集、验证集和测试集。
        3. 使用Optuna进行超参数优化。
        4. 使用找到的最佳超参数训练最终模型。
        5. 保存最终模型、配置和训练过程图表。
        6. (可选) 使用新训练的模型在内部测试集上进行评估。
        """
        self.logger.info("--- 开始执行模型训练全流程 ---")

        # 1. 加载和预处理数据 (fit_scalers=True 表示会拟合和保存缩放器)
        df_processed = self._load_and_preprocess_data_core(self.config['file_path'], fit_scalers=True)
        # 获取输入特征的数量，这是从预处理步骤中确定的 self.all_feature_columns_for_sequence 的长度
        num_input_features_for_model = len(self.all_feature_columns_for_sequence)
        if num_input_features_for_model == 0:
            self.logger.error("错误: 预处理后没有有效的输入特征列。终止训练。")
            return

        # 2. 创建输入/输出序列
        X_initial, y_initial = create_sequences(
            df_processed, self.config['look_back'], self.config['horizon'],
            self.config['target_col_names'], self.all_feature_columns_for_sequence,
            logger=self.logger
        )
        if X_initial.size == 0 or y_initial.size == 0: # 如果序列为空
            self.logger.error("错误: 创建输入/输出序列后数据为空。终止训练。")
            return

        # 3. 数据集划分 (70% 训练, 15% 验证, 15% 测试)
        total_samples = X_initial.shape[0] # 总样本数
        train_idx_end = int(total_samples * 0.7) # 训练集结束索引
        val_idx_end = int(total_samples * 0.85)  # 验证集结束索引 (训练集+验证集)

        X_train_np, y_train_np = X_initial[:train_idx_end], y_initial[:train_idx_end]
        X_val_np, y_val_np = X_initial[train_idx_end:val_idx_end], y_initial[train_idx_end:val_idx_end]
        X_test_np, y_test_np = X_initial[val_idx_end:], y_initial[val_idx_end:] # 测试集是剩余部分

        self.logger.info(f"数据集划分完毕: 训练集样本数={X_train_np.shape[0]}, 验证集样本数={X_val_np.shape[0]}, 测试集样本数={X_test_np.shape[0]}")

        if X_train_np.shape[0] == 0 or X_val_np.shape[0] == 0: # 确保训练集和验证集不为空
            self.logger.error("错误: 训练集或验证集在划分后为空。无法继续训练。")
            return

        # 4. Optuna超参数优化
        self.logger.info("\n--- 开始Optuna超参数优化 ---")
        study = optuna.create_study(
            direction='minimize', # 优化目标是最小化验证损失
            # 使用HyperbandPruner进行剪枝，可以提前终止不佳的试验
            pruner=optuna.pruners.HyperbandPruner(min_resource=5, max_resource=self.config['optuna_epochs'], reduction_factor=3),
            sampler=optuna.samplers.TPESampler(seed=42) # 使用TPE采样器，并设置种子以保证可复现性
        )
        # 执行优化，调用_objective_optuna函数，n_trials是试验次数，timeout是最大优化时间
        study.optimize(
            lambda trial: self._objective_optuna(trial, X_train_np, y_train_np, X_val_np, y_val_np, num_input_features_for_model),
            n_trials=self.config['n_optuna_trials'], timeout=3600*6 # 例如，6小时超时
        )
        best_hyperparams = study.best_params # 获取最佳超参数组合
        self.logger.info(f"Optuna超参数优化完成。")
        self.logger.info(f"最佳试验的验证损失 ({self.config['primary_target_col_name']}): {study.best_value:.6f}")
        self.logger.info(f"找到的最佳超参数组合: {best_hyperparams}")

        # 5. 使用最佳超参数训练最终模型
        self.logger.info("\n--- 使用最佳超参数训练最终模型 ---")
        # 从最佳超参数中提取模型架构相关的参数
        final_model_arch_params = {
            'd_model': best_hyperparams['d_model'], 'nhead': best_hyperparams['num_heads'],
            'num_encoder_layers': best_hyperparams['num_encoder_layers'],
            'dim_feedforward': best_hyperparams['d_model'] * best_hyperparams['dim_feedforward_factor'], # 注意这里要计算
            'dropout': best_hyperparams['dropout_rate'], 'norm_first': best_hyperparams['norm_first']
        }
        # 构建最终模型
        final_model = AQITransformer(
            num_features=num_input_features_for_model, **final_model_arch_params,
            horizon=self.config['horizon'], num_target_features=len(self.config['target_col_names'])
        ).to(DEVICE)

        # 创建最终训练和验证的数据加载器
        final_train_loader = DataLoader(TimeSeriesDataset(X_train_np, y_train_np), batch_size=self.config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
        final_val_loader = DataLoader(TimeSeriesDataset(X_val_np, y_val_np), batch_size=self.config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

        # 创建最终优化器和学习率调度器
        final_optimizer = torch.optim.AdamW(
            final_model.parameters(), lr=best_hyperparams['learning_rate'],
            weight_decay=best_hyperparams.get('weight_decay', 0.0) # 使用get以防optuna未优化此参数
        )
        final_scheduler = ReduceLROnPlateau(
            final_optimizer, mode='min', factor=best_hyperparams.get('scheduler_factor', 0.5),
            patience=best_hyperparams.get('scheduler_patience', 7)
        )
        criterion = nn.MSELoss() # 损失函数

        # 调用核心训练循环进行最终模型训练
        final_model, train_losses, val_losses, _ = self._train_model_core(
            final_model, final_train_loader, final_val_loader, criterion, final_optimizer, final_scheduler,
            epochs=self.config['full_train_epochs'] # 使用配置中定义的完整训练轮数
        )

        # 绘制并保存最终模型的训练损失图
        plot_training_loss(train_losses, val_losses,
                           os.path.join(self.config['model_artifacts_dir'], "final_model_training_loss.png"),
                           title_prefix="最终模型")
        self.logger.info(f"最终模型训练完成。最佳模型状态已保存。")

        # 6. 保存模型配置信息
        model_config_to_save = {
            'model_architecture': final_model_arch_params, # 模型架构参数
            'look_back': self.config['look_back'],
            'horizon': self.config['horizon'],
            'target_col_names': self.config['target_col_names'],
            'primary_target_col_name': self.config['primary_target_col_name'],
            'all_feature_columns_for_sequence': self.all_feature_columns_for_sequence, # 预处理后实际使用的特征列
            'num_input_features_for_model': num_input_features_for_model, # 输入特征数量
            'num_target_features': len(self.config['target_col_names']), # 目标特征数量
            'optuna_best_params': best_hyperparams, # Optuna找到的最佳超参数
            'enable_iqr_outlier_detection': self.config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION) # IQR检测状态
        }
        with open(os.path.join(self.config['model_artifacts_dir'], MODEL_CONFIG_SAVE_NAME), 'w', encoding='utf-8') as f:
            json.dump(model_config_to_save, f, indent=4, ensure_ascii=False) # ensure_ascii=False 保证中文正常显示
        self.logger.info(f"模型配置信息已保存至 {MODEL_CONFIG_SAVE_NAME}。")

        # 7. (可选) 在内部测试集上评估新训练的模型
        if X_test_np.shape[0] > 0: # 如果测试集不为空
            self.logger.info("开始使用新训练的模型在内部测试集上进行评估...")
            evaluator = ModelEvaluator( # 创建评估器实例
                model=final_model, # 传入训练好的模型
                model_config=model_config_to_save, # 模型配置
                feature_scaler=self.feature_scaler, # 特征缩放器 (已在预处理中拟合和保存)
                target_scalers=self.target_scalers, # 目标缩放器 (已在预处理中拟合和保存)
                device=DEVICE,
                logger=self.logger
            )
            # 注意：这里的y_test_np是原始（未缩放）的目标值，因为create_sequences返回的是原始目标值
            # 而X_test_np是缩放后的特征值。ModelEvaluator内部会处理目标值的反向缩放。
            # 但这里传入的y_test_np是已经经过create_sequences处理的，其目标值部分是原始的，
            # 而ModelEvaluator.evaluate_from_prepared_sequences 期望的y_test_np_original_targets就是这个原始值。
            evaluator.evaluate_from_prepared_sequences(X_test_np, y_test_np)
        else:
            self.logger.info("内部测试集为空，跳过最终模型评估步骤。")

class ModelPredictor:
    """
    模型预测器类，用于加载已训练的模型和相关组件（配置、缩放器），
    并对新的输入数据进行预测。
    """
    def __init__(self, artifacts_dir, logger):
        """
        初始化ModelPredictor。
        参数:
            artifacts_dir (str): 存放已训练模型、配置、缩放器等文件的目录路径。
            logger (logging.Logger): 日志记录器实例。
        """
        self.artifacts_dir = artifacts_dir
        self.logger = logger
        self.model = None # 加载后的PyTorch模型
        self.feature_scaler = None # 加载后的特征缩放器
        self.target_scalers = {} # 加载后的目标变量缩放器字典
        self.model_config = None # 加载后的模型配置字典
        self._load_artifacts() # 在初始化时即加载所有必要的组件

    def _load_artifacts(self):
        """
        从指定的目录加载模型状态、模型配置、特征缩放器和目标缩放器。
        """
        self.logger.info(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (ModelPredictor)...")
        # 构建各个文件的完整路径
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)

        # 检查所有必要文件是否存在
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing_files = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing_files}。")

        # 加载模型配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)

        # 加载特征缩放器和目标缩放器
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)

        # 根据配置构建模型骨架
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'],
            **self.model_config['model_architecture'], # 使用配置中的模型架构参数
            horizon=self.model_config['horizon'],
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE) # 模型移至计算设备

        # 加载模型权重
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval() # 设置为评估模式，因为是用于预测
        self.logger.info("模型、配置及缩放器已成功加载 (ModelPredictor)。")

    def _preprocess_input_for_prediction(self, df_raw):
        """
        对原始输入数据进行预处理，使其符合模型预测的要求。
        此预处理流程应与训练时的预处理流程（特征工程部分）保持一致。
        主要步骤包括：
        1. 时间戳解析和索引设置。
        2. 数值类型转换和NaN填充（特征列）。
        3. (如果需要) 计算AQI_calculated。
        4. 创建周期性时间特征。
        5. 创建滞后特征 (与训练时使用的滞后特征一致)。
        6. 确保所有模型期望的特征都存在，并按正确的顺序排列。
        7. 使用已加载的特征缩放器对特征进行缩放。
        参数:
            df_raw (pd.DataFrame): 原始输入数据DataFrame。
        返回:
            pd.DataFrame: 经过预处理和缩放后的特征DataFrame，可直接用于创建预测序列。
        """
        self.logger.info("开始为预测任务预处理输入数据 (ModelPredictor)...")
        df_processed = df_raw.copy() # 创建副本

        # 1. 时间戳处理 (与训练时逻辑类似)
        if not isinstance(df_processed.index, pd.DatetimeIndex): # 如果索引不是DatetimeIndex
            if 'Time' in df_processed.columns:
                df_processed['timestamp'] = pd.to_datetime(df_processed['Time'])
                df_processed = df_processed.set_index('timestamp').drop(columns=['Time'], errors='ignore')
            elif 'date' in df_processed.columns and 'hour' in df_processed.columns:
                 df_processed['datetime_str'] = df_processed['date'].astype(str) + df_processed['hour'].astype(int).astype(str).str.zfill(2)
                 df_processed['timestamp'] = pd.to_datetime(df_processed['datetime_str'], format='%Y%m%d%H')
                 df_processed = df_processed.set_index('timestamp').drop(columns=['date', 'hour', 'datetime_str'], errors='ignore')
            else: # 尝试将第一列作为时间索引
                try:
                    if len(df_processed.columns) > 0 and not pd.api.types.is_numeric_dtype(df_processed.iloc[:,0]):
                        df_processed.iloc[:, 0] = pd.to_datetime(df_processed.iloc[:, 0])
                        df_processed = df_processed.set_index(df_processed.columns[0])
                except Exception as e:
                    raise ValueError(f"错误: 无法为预测数据自动设置时间索引: {e}。")

        # 2. 数值转换和特征列NaN填充
        for col in df_processed.columns: # 遍历所有列
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # 获取模型配置中的目标列名，以排除它们，只处理特征列
        feature_cols_in_df = [col for col in df_processed.columns if col not in self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)]
        for col in feature_cols_in_df:
            if df_processed[col].isnull().any(): # 如果特征列有NaN
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0) # 前向、后向、0填充

        # 3. (如果需要) 计算AQI_calculated
        # 检查模型训练时是否使用了 'AQI_calculated' 作为特征
        if 'AQI_calculated' in self.model_config['all_feature_columns_for_sequence']:
            df_processed = calculate_aqi_from_pollutants(df_processed.copy()) # 计算AQI
            if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0) # 填充NaN
            elif 'AQI_calculated' not in df_processed.columns: # 如果计算失败或缺少列
                 self.logger.warning("预测数据中缺少计算'AQI_calculated'所需的污染物列或计算失败。将 'AQI_calculated' 特征填充为0。")
                 df_processed['AQI_calculated'] = 0

        # 4. 创建周期性时间特征
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
        else: # 如果索引不是时间类型，但模型期望这些周期特征，则填充为0
            expected_cyclical = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
            for feat_name in expected_cyclical:
                if feat_name in self.model_config['all_feature_columns_for_sequence'] and feat_name not in df_processed.columns:
                    self.logger.warning(f"预测数据的索引不是DatetimeIndex，但模型期望周期特征 '{feat_name}'。将填充为0。")
                    df_processed[feat_name] = 0

        # 5. 创建滞后特征 (与训练时使用的滞后特征一致)
        # 从模型配置中获取训练时使用的所有滞后特征名
        lag_features_to_recreate = [f for f in self.model_config['all_feature_columns_for_sequence'] if "_lag_" in f]
        lag_series_list_pred = []
        for lag_col_name in lag_features_to_recreate: # 遍历每个期望的滞后特征
            original_col_name_parts = lag_col_name.split("_lag_") # 解析出原始列名和滞后阶数
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]
                lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns: # 如果原始列存在于当前数据中
                    lag_series_list_pred.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else: # 如果原始列不存在，则无法创建此滞后特征，填充为0
                    self.logger.warning(f"警告: 创建滞后特征'{lag_col_name}'所需的原始列'{original_col}'在预测数据中未找到。该滞后特征将填充为0。")
                    lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else: # 如果滞后特征名格式不正确
                self.logger.warning(f"警告: 滞后特征名 '{lag_col_name}' 格式无法解析。该特征将填充为0。")
                lag_series_list_pred.append(pd.Series(0, index=df_processed.index, name=lag_col_name))

        if lag_series_list_pred: # 如果生成了任何滞后特征
             df_processed = pd.concat([df_processed] + lag_series_list_pred, axis=1)
        # 注意：这里创建滞后特征后，序列开头的NaN值需要被后续的create_sequences处理，
        # 或者在取最后look_back个数据点时，确保这些数据点没有因滞后产生的NaN。
        # 通常，预测时我们会假设输入数据已经足够长，至少look_back个有效数据点。
        # 如果输入数据本身就包含NaN，应该在之前步骤填充。滞后产生的NaN在序列最前面，
        # 如果我们只取最后一段数据进行预测，这些NaN可能不影响。

        # 6. 确保所有模型期望的特征都存在，并按正确的顺序排列
        features_dict = {} # 用于构建最终特征DataFrame的字典
        expected_features_from_config = self.model_config['all_feature_columns_for_sequence'] # 从配置中获取期望的特征列表
        for f_col in expected_features_from_config:
            if f_col in df_processed.columns: # 如果特征在处理后的数据中
                series_data = pd.to_numeric(df_processed[f_col], errors='coerce').fillna(0) # 再次确保是数值并填充0
                if not pd.api.types.is_numeric_dtype(series_data): # 最终检查
                    self.logger.error(f"预测预处理中，特征列 '{f_col}' 无法可靠转换为数值类型。")
                    try: features_dict[f_col] = series_data.astype(float)
                    except ValueError: raise ValueError(f"无法将特征列 '{f_col}' 转换为浮点数。")
                else: features_dict[f_col] = series_data
            else: # 如果模型期望的特征在处理后仍然缺失
                self.logger.warning(f"模型期望的特征列 '{f_col}' 在预测数据预处理后仍缺失。将创建为全0列。")
                features_dict[f_col] = pd.Series(0, index=df_processed.index, name=f_col) # 创建全0列

        # 构建包含所有期望特征的DataFrame，并按配置中的顺序排列
        df_for_scaling = pd.DataFrame(features_dict, index=df_processed.index)
        df_for_scaling = df_for_scaling[expected_features_from_config] # 确保列序正确

        # 7. 使用已加载的特征缩放器进行缩放
        # transform() 方法期望输入与fit()时具有相同的特征数量和顺序
        df_for_scaling_transformed = self.feature_scaler.transform(df_for_scaling)
        # 将缩放后的numpy数组转换回DataFrame，保持索引和列名
        df_scaled_features = pd.DataFrame(df_for_scaling_transformed, columns=expected_features_from_config, index=df_for_scaling.index)

        self.logger.info("预测输入数据预处理完成。")
        return df_scaled_features

    def predict(self, input_data_path_or_df):
        """
        使用加载的模型对输入数据进行预测。
        参数:
            input_data_path_or_df (str or pd.DataFrame):
                - 如果是str: CSV或Excel文件的路径。
                - 如果是pd.DataFrame: 包含输入特征的DataFrame。
                  此DataFrame应包含模型训练时使用的所有原始特征（在进行特征工程和缩放之前）。
                  其长度应至少为 look_back。
        返回:
            tuple: (predictions_original_all_targets, last_known_timestamp)
                - predictions_original_all_targets (np.array or None):
                    预测结果，形状为 (horizon, num_target_features)，已反向缩放回原始尺度。
                    如果预测失败，则为None。
                - last_known_timestamp (pd.Timestamp or None):
                    从输入数据中解析出的最后一个有效时间戳。如果无法解析，则为None。
        """
        # 1. 加载原始数据
        if isinstance(input_data_path_or_df, str): # 如果输入是文件路径
            try:
                df_raw = pd.read_csv(input_data_path_or_df)
            except Exception: # 尝试读取Excel
                try:
                    df_raw = pd.read_excel(input_data_path_or_df)
                except FileNotFoundError:
                    raise FileNotFoundError(f"错误: 预测数据文件 '{input_data_path_or_df}' 未找到。")
                except Exception as e_excel:
                    raise ValueError(f"错误: 读取Excel预测数据文件 '{input_data_path_or_df}' 失败: {e_excel}")
        elif isinstance(input_data_path_or_df, pd.DataFrame): # 如果输入是DataFrame
            df_raw = input_data_path_or_df.copy()
        else:
            raise ValueError("错误: 输入数据必须是CSV/Excel文件路径或Pandas DataFrame。")

        # 尝试获取输入数据的最后一个时间戳，用于后续生成预测结果的时间标签
        last_known_timestamp = None
        if isinstance(df_raw.index, pd.DatetimeIndex) and not df_raw.empty:
            last_known_timestamp = df_raw.index[-1]
        elif not df_raw.empty: # 如果索引不是DatetimeIndex，尝试从列中解析
            time_cols_to_check = ['Time', 'timestamp', 'Datetime'] # 常见的实际时间列名
            for tc in time_cols_to_check:
                if tc in df_raw.columns:
                    try: last_known_timestamp = pd.to_datetime(df_raw[tc].iloc[-1]); break
                    except: pass # 解析失败则尝试下一个
            if last_known_timestamp is None and 'date' in df_raw.columns and 'hour' in df_raw.columns: # 尝试合并date和hour
                try:
                    date_val = str(df_raw['date'].iloc[-1])
                    hour_val = str(int(df_raw['hour'].iloc[-1])).zfill(2)
                    last_dt_str = date_val + hour_val
                    last_known_timestamp = pd.to_datetime(last_dt_str, format='%Y%m%d%H')
                except: pass
            elif last_known_timestamp is None: # 尝试将第一列作为时间
                 try:
                     if not pd.api.types.is_numeric_dtype(df_raw.iloc[:,0]): # 确保不是纯数字列
                         last_known_timestamp = pd.to_datetime(df_raw.iloc[-1, 0])
                 except: pass
        if last_known_timestamp is None:
            self.logger.warning("警告: 无法从输入数据中可靠地确定最后一个时间戳。预测结果的时间标签可能不准确。")

        # 2. 预处理输入数据 (特征工程和缩放)
        df_processed_features = self._preprocess_input_for_prediction(df_raw.copy())

        # 3. 检查预处理后的数据长度是否满足look_back要求
        if len(df_processed_features) < self.model_config['look_back']:
            raise ValueError(f"错误: 经过预处理后的预测输入数据长度为 {len(df_processed_features)}，"
                             f"不足以满足模型所需的回溯窗口长度 {self.model_config['look_back']}。")

        # 4. 准备模型输入序列
        # 从模型配置中获取训练时使用的所有特征列名
        model_input_feature_names = self.model_config['all_feature_columns_for_sequence']
        # 确保所有这些特征都存在于预处理后的数据中，并按正确顺序排列
        available_model_features = [col for col in model_input_feature_names if col in df_processed_features.columns]
        if len(available_model_features) != len(model_input_feature_names):
            missing_for_pred = set(model_input_feature_names) - set(available_model_features)
            self.logger.warning(f"警告: 预处理后，部分模型期望的特征列在最终输入数据中缺失: {missing_for_pred}。这可能导致预测不准确。")
            # 注意：_preprocess_input_for_prediction 应该已经处理了缺失特征（填充为0），
            # 所以这里主要是为了日志记录和潜在的调试。

        # 截取最后 look_back 个时间步的数据作为模型输入
        # df_processed_features 已经是按 model_input_feature_names 顺序排列的
        last_input_sequence_df = df_processed_features[available_model_features].iloc[-self.model_config['look_back']:]
        X_pred_np = np.array([last_input_sequence_df.values]) # 转换为 (1, look_back, num_features) 的numpy数组

        if X_pred_np.size == 0: # 如果未能创建输入序列
            self.logger.error("错误: 无法创建用于预测的输入序列 (X_pred_np为空)。")
            return None, last_known_timestamp

        X_pred_torch = torch.from_numpy(X_pred_np).float().to(DEVICE) # 转换为PyTorch张量并移至设备

        # 5. 模型预测
        with torch.no_grad(): # 不计算梯度
            predictions_scaled = self.model(X_pred_torch) # (1, horizon, num_target_features)

        # 6. 反向缩放预测结果
        # predictions_scaled 是 (1, horizon, num_target_features)，取第一个样本的预测
        predictions_scaled_np_slice = predictions_scaled.cpu().numpy()[0, :, :] # (horizon, num_target_features)
        predictions_original_all_targets = np.zeros_like(predictions_scaled_np_slice) # 初始化用于存储原始尺度预测的数组

        for i, col_name in enumerate(self.model_config['target_col_names']): # 遍历每个目标变量
            scaler = self.target_scalers[col_name] # 获取对应的缩放器
            # 对该目标变量在所有horizon步上的预测值进行反向缩放
            pred_col_scaled_reshaped = predictions_scaled_np_slice[:, i].reshape(-1, 1) # (horizon, 1)
            predictions_original_all_targets[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten() # (horizon,)

        self.logger.info(f"已成功生成 {self.model_config['horizon']} 个时间步的预测。")
        return predictions_original_all_targets, last_known_timestamp

class ModelEvaluator:
    """
    模型评估器类，用于评估已训练模型的性能。
    它可以从原始数据文件或已准备好的序列数据进行评估。
    """
    def __init__(self, model, model_config, feature_scaler, target_scalers, device, logger):
        """
        初始化ModelEvaluator。
        参数:
            model (nn.Module): 已加载或已训练的PyTorch模型。
            model_config (dict): 模型的配置信息。
            feature_scaler (StandardScaler): 已加载或已拟合的特征缩放器。
            target_scalers (dict): 包含各目标变量缩放器的字典。
            device (torch.device): 计算设备 (CPU或GPU)。
            logger (logging.Logger): 日志记录器实例。
        """
        self.model = model
        self.model_config = model_config
        self.feature_scaler = feature_scaler
        self.target_scalers = target_scalers
        self.device = device
        self.logger = logger

        # 从配置中提取常用参数
        self.all_feature_columns_for_sequence = model_config['all_feature_columns_for_sequence']
        self.target_col_names = model_config['target_col_names']
        self.look_back = model_config['look_back']
        self.horizon = model_config['horizon']
        self.batch_size = model_config.get('batch_size', DEFAULT_BATCH_SIZE) # 如果配置中没有，使用默认值

    def _preprocess_evaluation_data(self, df_raw):
        """
        为评估任务预处理输入数据。此流程与训练时的预处理（特征工程部分）和
        预测时的预处理（_preprocess_input_for_prediction）非常相似，
        主要区别在于评估数据必须包含真实的目标值。
        参数:
            df_raw (pd.DataFrame): 包含原始特征和目标值的DataFrame。
        返回:
            pd.DataFrame: 经过预处理的DataFrame，其中特征已缩放，目标值保持原始尺度。
                          此DataFrame可直接用于create_sequences。
        """
        self.logger.info("开始为评估任务预处理输入数据 (ModelEvaluator)...")
        df_processed = df_raw.copy()

        # 1. 时间戳处理 (与训练/预测时逻辑一致)
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
                    raise ValueError(f"错误: 无法为评估数据自动设置时间索引: {e}")

        # 2. 数值转换
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 3. 检查和处理目标列中的NaN (评估数据必须有有效的真实目标值)
        for tc in self.target_col_names:
            if tc in df_processed.columns:
                if df_processed[tc].isnull().any():
                    self.logger.warning(f"目标列 '{tc}' 在评估数据中包含NaN值。将尝试删除包含NaN的行。")
                    df_processed.dropna(subset=[tc], inplace=True) # 删除包含NaN的行
            else: # 如果目标列缺失
                raise ValueError(f"评估错误: 目标列 '{tc}' 在评估数据中缺失。无法进行评估。")
        if df_processed.empty: # 如果处理后数据为空
            raise ValueError("评估错误: 处理目标列中的NaN后，评估数据为空。")

        # 4. 特征列NaN填充 (与训练/预测时逻辑一致)
        feature_candidate_cols = [col for col in df_processed.columns if col not in self.target_col_names]
        for col in feature_candidate_cols:
            if col in df_processed.columns and df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)

        # 5. (如果需要) 计算AQI_calculated (与训练/预测时逻辑一致)
        if 'AQI_calculated' in self.all_feature_columns_for_sequence:
            df_processed = calculate_aqi_from_pollutants(df_processed.copy())
            if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            elif 'AQI_calculated' not in df_processed.columns:
                 df_processed['AQI_calculated'] = 0 # 如果计算失败或缺少列

        # 6. 创建周期性时间特征 (与训练/预测时逻辑一致)
        new_cyclical_features = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            idx = df_processed.index
            new_cyclical_features['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24.0)
            # ... (省略其他周期特征的重复代码，与_preprocess_input_for_prediction中一致)
            new_cyclical_features['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['dayofweek_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
            new_cyclical_features['month_sin'] = np.sin(2 * np.pi * idx.month / 12.0)
            new_cyclical_features['month_cos'] = np.cos(2 * np.pi * idx.month / 12.0)
            df_processed = pd.concat([df_processed, new_cyclical_features], axis=1)
        else: # 如果索引不是时间类型，但模型期望这些周期特征
            expected_cyclical_feats = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
            for feat in expected_cyclical_feats:
                if feat in self.all_feature_columns_for_sequence and feat not in df_processed.columns:
                    self.logger.warning(f"评估数据的索引不是DatetimeIndex，但模型期望周期特征 '{feat}'。将填充为0。")
                    df_processed[feat] = 0

        # 7. 创建滞后特征 (与训练/预测时逻辑一致)
        lag_features_to_concat = []
        lag_cols_created_names = []
        # 确定需要哪些原始列来创建模型期望的滞后特征
        base_cols_for_lags_needed = set()
        for f_col in self.all_feature_columns_for_sequence:
            if "_lag_" in f_col:
                base_cols_for_lags_needed.add(f_col.split("_lag_")[0])

        for original_col_name in base_cols_for_lags_needed:
            if original_col_name in df_processed.columns:
                # 确定该原始列需要创建的最大滞后阶数
                max_lag_for_this_col = 0
                for f_col in self.all_feature_columns_for_sequence:
                    if f_col.startswith(original_col_name + "_lag_"):
                        try: max_lag_for_this_col = max(max_lag_for_this_col, int(f_col.split("_lag_")[1]))
                        except: pass # 忽略无法解析的滞后名
                # 创建从1到max_lag_for_this_col的滞后特征
                for lag in range(1, max_lag_for_this_col + 1):
                    lag_col_name = f"{original_col_name}_lag_{lag}"
                    if lag_col_name in self.all_feature_columns_for_sequence: # 确保这个滞后特征是模型期望的
                        lag_features_to_concat.append(df_processed[original_col_name].shift(lag).rename(lag_col_name))
                        lag_cols_created_names.append(lag_col_name)
            else:
                self.logger.warning(f"创建滞后特征所需的原始列 '{original_col_name}' 在评估数据中未找到。")

        if lag_features_to_concat:
            df_processed = pd.concat([df_processed] + lag_features_to_concat, axis=1)
        if lag_cols_created_names: # 如果创建了滞后特征，删除因滞后产生的NaN行
            df_processed.dropna(subset=lag_cols_created_names, inplace=True)
        if df_processed.empty: # 如果处理后数据为空
            raise ValueError("评估错误: 创建滞后特征并处理相关NaN后，评估数据为空。")

        # 8. 确保所有模型期望的特征都存在，并按正确的顺序排列 (与预测时逻辑一致)
        features_dict = {}
        for f_col in self.all_feature_columns_for_sequence:
            if f_col in df_processed.columns:
                series_data = pd.to_numeric(df_processed[f_col], errors='coerce').fillna(0)
                if not pd.api.types.is_numeric_dtype(series_data):
                    self.logger.error(f"评估预处理中，特征列 '{f_col}' 无法可靠转换为数值类型。")
                    try: features_dict[f_col] = series_data.astype(float)
                    except ValueError: raise ValueError(f"无法将特征列 '{f_col}' 转换为浮点数。")
                else: features_dict[f_col] = series_data
            else:
                self.logger.warning(f"模型期望的特征列 '{f_col}' 在评估数据特征工程后缺失。将创建为全0列。")
                features_dict[f_col] = pd.Series(0, index=df_processed.index, name=f_col)

        df_features_to_scale = pd.DataFrame(features_dict, index=df_processed.index)
        df_features_to_scale = df_features_to_scale[self.all_feature_columns_for_sequence] # 保证列序

        # 9. 使用已加载的特征缩放器对特征进行缩放
        scaled_features_np = self.feature_scaler.transform(df_features_to_scale)
        scaled_features_df = pd.DataFrame(scaled_features_np, columns=self.all_feature_columns_for_sequence, index=df_features_to_scale.index)

        # 10. 将原始尺度的目标值合并回DataFrame，用于后续create_sequences
        df_for_sequences = scaled_features_df.copy()
        for tc in self.target_col_names:
            if tc in df_processed.columns: # df_processed 中的目标列是原始尺度的
                df_for_sequences[tc] = df_processed[tc]
            else: # 这不应该发生，因为前面已经检查过目标列
                raise ValueError(f"严重错误: 目标列 '{tc}' 在准备序列创建时从评估数据中丢失。")

        self.logger.info("评估数据预处理完成。")
        return df_for_sequences # 返回的DataFrame中，特征是缩放的，目标是原始的

    def _calculate_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差 (MAPE)。"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # 避免除以零，只对真实值非零（或接近零）的部分计算MAPE
        non_zero_mask = np.abs(y_true) > 1e-9 # 使用一个小的epsilon来判断是否接近零
        if np.sum(non_zero_mask) == 0: return np.nan # 如果所有真实值都接近零，无法计算MAPE
        return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

    def _calculate_smape(self, y_true, y_pred):
        """计算对称平均绝对百分比误差 (SMAPE)。"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        epsilon = 1e-9 # 一个小值，防止分母为零
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        # 如果分母接近零，则该点的SMAPE贡献为0（避免除以零）
        smape = np.where(denominator < epsilon, 0.0, numerator / denominator) * 100
        return np.mean(smape)

    def evaluate_from_prepared_sequences(self, X_test_np, y_test_np_original_targets):
        """
        使用已经预处理和序列化好的数据 (X_test_np, y_test_np_original_targets) 进行模型评估。
        参数:
            X_test_np (np.array): 测试集的输入序列，特征已缩放。形状 (样本数, look_back, 特征数)。
            y_test_np_original_targets (np.array): 测试集的目标序列，目标值保持原始尺度。
                                                  形状 (样本数, horizon, 目标数)。
        返回:
            dict: 包含各目标变量评估指标的字典。
                  键为目标变量名，值为包含 'MAE', 'RMSE', 'R2', 'MAPE', 'SMAPE' 的字典。
        """
        self.logger.info("开始使用预处理和序列化的数据进行模型评估...")
        if X_test_np.size == 0 or y_test_np_original_targets.size == 0: # 检查数据是否为空
            self.logger.error("提供的序列数据为空。评估中止。")
            return {}

        # 1. 创建测试数据加载器 (只需要X，因为模型直接预测，然后与y比较)
        test_dataset = TensorDataset(torch.from_numpy(X_test_np).float()) # 只包含X
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # 2. 模型预测
        self.model.eval() # 设置为评估模式
        all_preds_scaled_list = [] # 存储所有批次的缩放后预测结果
        with torch.no_grad():
            for x_batch_tuple in test_loader: # 遍历测试数据
                x_batch = x_batch_tuple[0].to(self.device) # DataLoader返回的是元组，取第一个元素
                outputs_scaled = self.model(x_batch) # 模型预测 (输出是缩放后的)
                all_preds_scaled_list.append(outputs_scaled.cpu().numpy()) # 转到CPU并转为numpy

        if not all_preds_scaled_list: # 如果没有预测结果
            self.logger.error("模型未能对测试序列生成任何预测。评估中止。")
            return {}
        preds_scaled_np = np.concatenate(all_preds_scaled_list, axis=0) # (样本数, horizon, 目标数)

        # 3. 反向缩放预测结果
        predicted_orig_all_targets = np.zeros_like(preds_scaled_np) # 初始化原始尺度预测数组
        for i, col_name in enumerate(self.target_col_names): # 遍历每个目标
            scaler = self.target_scalers[col_name] # 获取对应缩放器
            for h_step in range(self.horizon): # 遍历预测的每个时间步
                 # 对每个目标、每个时间步的预测值进行反向缩放
                 predicted_orig_all_targets[:, h_step, i] = scaler.inverse_transform(preds_scaled_np[:, h_step, i].reshape(-1,1)).flatten()

        # 4. 准备实际目标值 (已经是原始尺度)
        actual_orig_all_targets = y_test_np_original_targets

        # 5. 计算评估指标
        metrics_results = {} # 存储所有指标结果
        self.logger.info("\n各目标污染物在测试集上的评估指标 (原始尺度):")
        for i, col_name in enumerate(self.target_col_names): # 遍历每个目标
            # 将该目标的实际值和预测值展平，以便计算指标
            # actual_orig_all_targets[:, :, i] 的形状是 (样本数, horizon)
            actual_col_flat = actual_orig_all_targets[:, :, i].flatten()
            predicted_col_flat = predicted_orig_all_targets[:, :, i].flatten()

            if len(actual_col_flat) == 0: # 如果没有有效数据
                self.logger.warning(f"目标 '{col_name}' 没有有效的扁平化数据进行评估。")
                metrics_results[col_name] = {'MAE': np.nan, 'RMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan, 'SMAPE': np.nan}
                continue

            # 计算各项指标
            mae = mean_absolute_error(actual_col_flat, predicted_col_flat)
            rmse = np.sqrt(mean_squared_error(actual_col_flat, predicted_col_flat))
            try: # R2计算可能因实际值无方差而出错
                r2 = r2_score(actual_col_flat, predicted_col_flat)
            except ValueError:
                r2 = np.nan
                self.logger.warning(f"R2计算失败，目标 '{col_name}' 的实际值可能没有方差。")
            mape = self._calculate_mape(actual_col_flat, predicted_col_flat)
            smape = self._calculate_smape(actual_col_flat, predicted_col_flat)

            metrics_results[col_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'SMAPE': smape}
            self.logger.info(f"  {col_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}, MAPE={mape:.2f}%, SMAPE={smape:.2f}%")

            # 绘制并保存该目标的实际值与预测值对比图
            plot_save_prefix = os.path.join(self.model_config.get('model_artifacts_dir', MODEL_ARTIFACTS_DIR), f"evaluation_prepared_seq_{col_name}")
            plot_predictions_vs_actual(
                actual_orig_all_targets[:, :, i], # 传入 (样本数, horizon) 形状的数据
                predicted_orig_all_targets[:, :, i],
                col_name,
                plot_save_prefix,
                title_suffix="测试集评估 (预处理序列)：实际值 vs. 预测值"
            )
        self.logger.info(f"测试集评估图表已保存至目录: {self.model_config.get('model_artifacts_dir', MODEL_ARTIFACTS_DIR)}")
        return metrics_results

    def evaluate_model_from_source(self, test_data_path_or_df):
        """
        从原始数据源（文件或DataFrame）评估模型。
        此方法会先调用_preprocess_evaluation_data进行数据预处理，
        然后调用create_sequences生成序列，最后调用evaluate_from_prepared_sequences进行评估。
        参数:
            test_data_path_or_df (str or pd.DataFrame): 测试数据的文件路径或DataFrame。
                                                       应包含原始特征和真实的原始尺度目标值。
        返回:
            dict: 包含各目标变量评估指标的字典。
        """
        self.logger.info(f"开始从源数据评估模型: {test_data_path_or_df if isinstance(test_data_path_or_df, str) else 'DataFrame'}")
        # 1. 加载原始数据
        if isinstance(test_data_path_or_df, str):
            try:
                df_raw = pd.read_csv(test_data_path_or_df)
            except Exception: # 尝试Excel
                df_raw = pd.read_excel(test_data_path_or_df)
        else:
            df_raw = test_data_path_or_df.copy()

        # 2. 预处理评估数据 (特征工程、特征缩放，目标保持原始)
        df_for_sequences = self._preprocess_evaluation_data(df_raw)

        # 3. 创建序列 (X是缩放特征，y是原始目标)
        X_test_np, y_test_np_original_targets = create_sequences(
            df_for_sequences, self.look_back, self.horizon,
            self.target_col_names, self.all_feature_columns_for_sequence,
            is_predict=False, logger=self.logger # is_predict=False 表示需要生成y
        )
        # 4. 使用准备好的序列进行评估
        return self.evaluate_from_prepared_sequences(X_test_np, y_test_np_original_targets)

class AQISystem:
    """
    空气质量指数（AQI）预测与异常检测系统的主类。
    封装了模型训练、预测、评估和异常检测的完整功能。
    """
    def __init__(self, artifacts_dir=MODEL_ARTIFACTS_DIR, config_overrides=None):
        """
        初始化AQISystem。
        参数:
            artifacts_dir (str, optional): 模型相关文件（权重、配置、缩放器等）的存储和加载目录。
                                           默认为全局定义的MODEL_ARTIFACTS_DIR。
            config_overrides (dict, optional): 用于覆盖默认配置参数的字典。默认为None。
        """
        self.artifacts_dir = artifacts_dir # 模型工件目录
        self.config = self._load_default_config() # 加载默认配置
        if config_overrides: # 如果有提供覆盖配置
            self.config.update(config_overrides) # 更新配置
        self.config['model_artifacts_dir'] = self.artifacts_dir # 确保配置中的工件目录与实例一致
        os.makedirs(self.artifacts_dir, exist_ok=True) # 创建工件目录（如果不存在）

        # 获取主日志记录器 (应由 setup_global_logging_and_redirect 在主程序入口处配置好)
        self.logger = logging.getLogger("AQIMainApp")
        set_seed() # 设置随机种子

        self.trainer = None # 模型训练器实例
        self.predictor_instance = None # 模型预测器实例（与AQISystem内部加载的model/scalers区分）
        self.model_config = None # 加载的模型配置
        self.feature_scaler = None # 加载的特征缩放器
        self.target_scalers = None # 加载的目标缩放器
        self.model = None # 加载的PyTorch模型
        self.all_feature_columns_for_sequence = None # 加载的特征列列表

    def _load_default_config(self):
        """加载系统的默认配置参数。"""
        return {
            'file_path': DEFAULT_FILE_PATH,
            'look_back': DEFAULT_LOOK_BACK,
            'horizon': DEFAULT_HORIZON,
            'target_col_names': DEFAULT_TARGET_COL_NAMES,
            'primary_target_col_name': DEFAULT_PRIMARY_TARGET_COL_NAME,
            'batch_size': DEFAULT_BATCH_SIZE,
            'model_artifacts_dir': self.artifacts_dir, # 确保这里也使用实例的artifacts_dir
            'full_train_epochs': DEFAULT_FULL_TRAIN_EPOCHS,
            'n_optuna_trials': DEFAULT_N_OPTUNA_TRIALS,
            'optuna_epochs': DEFAULT_OPTUNA_EPOCHS,
            'early_stopping_patience': DEFAULT_EARLY_STOPPING_PATIENCE,
            'min_delta': DEFAULT_MIN_DELTA,
            'anomaly_threshold_factor': DEFAULT_ANOMALY_THRESHOLD_FACTOR,
            'enable_iqr_outlier_detection': DEFAULT_ENABLE_IQR_OUTLIER_DETECTION
        }

    def _ensure_model_loaded_for_use(self):
        """
        确保模型及其相关组件（配置、缩放器）已被加载到AQISystem实例的属性中。
        如果尚未加载，则从工件目录加载它们。
        此方法主要供预测、评估、异常检测等需要已训练模型的功能调用。
        """
        # 如果所有必要组件都已加载，则直接返回
        if self.model is not None and self.model_config is not None and \
           self.feature_scaler is not None and self.target_scalers is not None and \
           self.all_feature_columns_for_sequence is not None:
            return

        self.logger.info(f"开始从目录 '{self.artifacts_dir}' 加载模型及相关组件 (AQISystem集中管理)...")
        # 构建文件路径
        config_path = os.path.join(self.artifacts_dir, MODEL_CONFIG_SAVE_NAME)
        model_path = os.path.join(self.artifacts_dir, MODEL_STATE_SAVE_NAME)
        fs_path = os.path.join(self.artifacts_dir, FEATURE_SCALER_SAVE_NAME)
        ts_path = os.path.join(self.artifacts_dir, TARGET_SCALERS_SAVE_NAME)

        # 检查文件是否存在
        required_files = [config_path, model_path, fs_path, ts_path]
        if not all(os.path.exists(p) for p in required_files):
            missing = [p for p in required_files if not os.path.exists(p)]
            raise FileNotFoundError(f"错误: 一个或多个必要的模型文件在目录 '{self.artifacts_dir}' 中未找到: {missing}。")

        # 加载模型配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.model_config = json.load(f)

        # 使用加载的配置更新AQISystem实例的配置（如果需要）
        # 这确保了当使用已保存模型时，系统配置与模型训练时的配置一致
        self.config['look_back'] = self.model_config.get('look_back', self.config['look_back'])
        self.config['horizon'] = self.model_config.get('horizon', self.config['horizon'])
        self.config['target_col_names'] = self.model_config.get('target_col_names', self.config['target_col_names'])
        self.config['primary_target_col_name'] = self.model_config.get('primary_target_col_name', self.config['primary_target_col_name'])
        self.config['enable_iqr_outlier_detection'] = self.model_config.get('enable_iqr_outlier_detection', DEFAULT_ENABLE_IQR_OUTLIER_DETECTION)

        # 加载特征列列表
        self.all_feature_columns_for_sequence = self.model_config.get('all_feature_columns_for_sequence')
        if self.all_feature_columns_for_sequence is None: # 必须有这个配置
            raise ValueError("错误: 加载的模型配置中未找到 'all_feature_columns_for_sequence'。")

        # 加载缩放器
        self.feature_scaler = joblib.load(fs_path)
        self.target_scalers = joblib.load(ts_path)

        # 构建模型并加载权重
        self.model = AQITransformer(
            num_features=self.model_config['num_input_features_for_model'],
            **self.model_config['model_architecture'],
            horizon=self.model_config['horizon'],
            num_target_features=self.model_config['num_target_features']
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval() # 设置为评估模式
        self.logger.info("模型、配置及缩放器已成功加载并准备就绪 (AQISystem集中管理)。")

    def train_new_model(self, train_data_path, enable_iqr_detection_override=None):
        """
        训练一个新的AQI预测模型。
        参数:
            train_data_path (str): 训练数据文件的路径。
            enable_iqr_detection_override (bool, optional): 是否覆盖配置中关于IQR异常检测的设置。
                                                           如果为None，则使用config中的设置。
                                                           如果为True/False，则强制使用该设置。
        """
        self.config['file_path'] = train_data_path # 更新配置文件路径
        if enable_iqr_detection_override is not None: # 如果提供了IQR覆盖设置
            self.config['enable_iqr_outlier_detection'] = enable_iqr_detection_override
            self.logger.info(f"信息: IQR异常值检测功能已通过命令行设置为: {'启用' if enable_iqr_detection_override else '禁用'}")

        # 为本次训练创建一个特定的日志文件记录器
        log_subdir = os.path.join(self.artifacts_dir, LOG_DIR_NAME) # 日志子目录
        os.makedirs(log_subdir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        training_log_file = os.path.join(log_subdir, f"training_specific_log_{timestamp}.log")

        # 获取或创建 Trainer 特定的记录器实例
        # 它会将日志消息传播到 AQIMainApp 记录器（因此输出到主控制台日志和控制台）
        # 并且它将有自己的文件处理器用于详细的训练日志。
        training_logger = get_configured_logger(name=f"Trainer_{timestamp}", log_file_path=training_log_file, level=logging.DEBUG)

        training_logger.info(f"开始新的模型训练流程。特定训练日志文件: {training_log_file}")
        training_logger.info(f"训练数据路径: {train_data_path}")
        training_logger.info(f"模型工件目录: {self.artifacts_dir}")
        training_logger.info(f"IQR检测状态: {'启用' if self.config['enable_iqr_outlier_detection'] else '禁用'}")

        self.trainer = ModelTrainer(self.config, training_logger) # 创建训练器实例
        try:
            self.trainer.run_training_pipeline() # 运行训练流程
            training_logger.info("模型训练完成。尝试加载新训练的模型组件到当前AQISystem实例...")
            # 训练完成后，尝试立即加载新训练的模型到AQISystem实例中，以便后续直接使用
            self._ensure_model_loaded_for_use()
        except Exception as e:
            training_logger.error(f"模型训练过程中发生严重错误: {e}", exc_info=True) # 记录详细错误信息

    def predict_with_existing_model(self, input_data_path_or_df):
        """
        使用已加载（或新训练）的模型进行预测。
        参数:
            input_data_path_or_df (str or pd.DataFrame): 输入数据的文件路径或DataFrame。
        """
        try:
            self._ensure_model_loaded_for_use() # 确保模型已加载

            # 如果还没有ModelPredictor实例，或者其工件目录与当前系统不一致，则创建一个新的
            # 这主要是为了封装预测逻辑，并确保使用的是与当前AQISystem实例匹配的工件
            if self.predictor_instance is None or self.predictor_instance.artifacts_dir != self.artifacts_dir:
                 self.predictor_instance = ModelPredictor(self.artifacts_dir, self.logger)
                 # 此时ModelPredictor内部会重新加载一次模型，虽然AQISystem也加载了。
                 # 可以优化为直接将AQISystem加载的model, scalers等传递给Predictor，
                 # 但当前结构下，Predictor独立加载保证了其封装性。

            # 调用ModelPredictor的predict方法
            predicted_values, last_timestamp = self.predictor_instance.predict(input_data_path_or_df)

            if predicted_values is not None: # 如果成功预测
                # 从预测器（或当前系统配置）获取horizon和目标名，用于打印结果
                horizon_from_pred_config = self.predictor_instance.model_config.get('horizon', self.config['horizon'])
                target_names_from_pred_config = self.predictor_instance.model_config.get('target_col_names', self.config['target_col_names'])

                self.logger.info(f"\n模型成功生成了未来 {horizon_from_pred_config} 小时的各项指标预测值 (原始尺度):")
                # 打印前5个（或所有，如果horizon<5）时间步的预测结果
                for h in range(min(5, horizon_from_pred_config)):
                    hour_str = f"  未来第 {h+1} 小时: "
                    for t_idx, t_name in enumerate(target_names_from_pred_config):
                        val_to_print = predicted_values[h, t_idx]
                        if t_name == 'CO': # CO保留两位小数
                            hour_str += f"{t_name}={val_to_print:.2f} "
                        else: # 其他指标四舍五入为整数
                            hour_str += f"{t_name}={np.round(val_to_print).astype(int)} "
                    self.logger.info(hour_str)

                # 询问用户是否保存详细预测结果到CSV
                if last_timestamp is not None: # 只有当能确定最后一个时间戳时才方便保存带时间的CSV
                    # 使用 self.logger.info 替代 print，让日志系统捕获它
                    self.logger.info("是否将详细预测结果保存到CSV文件? (y/n): (请在控制台输入)")
                    save_pred = input("是否将详细预测结果保存到CSV文件? (y/n): ").strip().lower() # input() 仍然直接与用户交互
                    self.logger.info(f"用户输入保存选项: {save_pred}") # 记录用户的选择

                    if save_pred.startswith('y'):
                        pred_save_path = os.path.join(self.artifacts_dir, "predictions_output_adv.csv")
                        # 生成未来的时间戳序列
                        future_timestamps = pd.date_range(
                            start=last_timestamp + pd.Timedelta(hours=1), # 从最后一个已知时间戳的下一个小时开始
                            periods=horizon_from_pred_config, freq='H' # 按小时频率，共horizon个
                        )
                        # 构建输出DataFrame的数据字典
                        output_data = {'date': future_timestamps.strftime('%Y%m%d'), 'hour': future_timestamps.hour}
                        for t_idx, t_name in enumerate(target_names_from_pred_config):
                            val_col = predicted_values[:, t_idx] # 该目标在所有horizon步上的预测值
                            if t_name == 'CO':
                                output_data[t_name] = np.round(val_col, 2)
                            else:
                                output_data[t_name] = np.round(val_col).astype(int)

                        # 定义期望输出的列顺序 (标准格式)
                        requested_output_columns = ['date', 'hour', 'AQI', 'PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']
                        final_output_df_cols = {}
                        for col in requested_output_columns: # 按期望顺序填充数据
                            # 如果模型预测了这个列，则使用预测值；否则填充NaN
                            final_output_df_cols[col] = output_data.get(col, [np.nan] * horizon_from_pred_config)

                        output_df = pd.DataFrame(final_output_df_cols)[requested_output_columns] # 创建DataFrame并按列序排列
                        output_df.to_csv(pred_save_path, index=False, encoding='utf-8-sig') # 保存到CSV (utf-8-sig确保Excel打开中文不乱码)
                        self.logger.info(f"预测结果已保存至: {pred_save_path}")

                        # 提示用户CSV中实际包含的预测列和未包含的列
                        predicted_cols_in_output = [col for col in target_names_from_pred_config if col in requested_output_columns]
                        nan_cols_in_output = [col for col in requested_output_columns if col not in ['date', 'hour'] and col not in predicted_cols_in_output]
                        self.logger.info(f"提示: CSV文件中实际包含预测数据的列为: {', '.join(predicted_cols_in_output)}。")
                        if nan_cols_in_output:
                            self.logger.info(f"以下在标准输出列中但未被当前模型预测的列，在CSV中已填充为NaN: {', '.join(nan_cols_in_output)}。")
                    else:
                        self.logger.info("预测结果未保存。")
                else: # 如果无法确定last_timestamp
                    self.logger.warning("由于无法从输入数据中确定最后一个有效时间戳，预测结果未关联具体日期，也无法保存为带时间戳的CSV文件。")
                    self.logger.info("原始预测值数组 (形状: horizon, num_targets):")
                    self.logger.info(np.round(predicted_values,2)) # 直接打印预测数组
            elif predicted_values is None: # 如果预测失败
                self.logger.error("模型未能成功生成预测。请检查日志或输入数据。")
        except FileNotFoundError as e:
             self.logger.error(f"文件未找到错误 (预测流程): {e}。请检查模型工件目录和输入文件路径。")
        except ValueError as e:
             self.logger.error(f"值错误 (预测流程): {e}")
        except Exception as e:
             self.logger.error(f"预测过程中发生未知错误: {e}", exc_info=True)

    def evaluate_existing_model(self, test_data_path_or_df):
        """
        评估已加载（或新训练）的模型的性能。
        参数:
            test_data_path_or_df (str or pd.DataFrame): 测试数据的文件路径或DataFrame。
                                                       应包含原始特征和真实的原始尺度目标值。
        """
        self.logger.info(f"开始评估现有模型。测试数据: {test_data_path_or_df if isinstance(test_data_path_or_df, str) else 'DataFrame'}")
        try:
            self._ensure_model_loaded_for_use() # 确保模型已加载
            # 创建评估器实例，传入AQISystem当前加载的模型和组件
            evaluator = ModelEvaluator(
                model=self.model, model_config=self.model_config,
                feature_scaler=self.feature_scaler, target_scalers=self.target_scalers,
                device=DEVICE, logger=self.logger
            )
            metrics = evaluator.evaluate_model_from_source(test_data_path_or_df) # 从源数据进行评估
            if metrics: self.logger.info("模型评估完成。详细指标已打印。")
            else: self.logger.error("模型评估未能生成结果。")
        except FileNotFoundError as e: self.logger.error(f"文件未找到错误 (评估流程): {e}", exc_info=True)
        except ValueError as e: self.logger.error(f"值错误 (评估流程): {e}", exc_info=True)
        except Exception as e: self.logger.error(f"评估过程中发生未知错误: {e}", exc_info=True)

    def _preprocess_input_for_anomaly(self, df_raw):
        """
        为异常检测任务预处理输入数据。
        此流程与预测和评估的预处理类似，但目标列的处理方式可能不同，
        因为异常检测时，我们关心的是模型对实际观测值的重建误差。
        参数:
            df_raw (pd.DataFrame): 原始输入数据DataFrame，应包含特征和实际目标值。
        返回:
            pd.DataFrame: 经过预处理的DataFrame，特征已缩放，目标值保持原始尺度。
                          可直接用于create_sequences进行异常检测。
        """
        self.logger.info("开始为异常检测任务预处理输入数据 (AQISystem)...")
        df_processed = df_raw.copy()

        # 1. 时间戳处理 (与预测/评估一致)
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            # ... (省略重复的时间戳解析代码，与_preprocess_input_for_prediction中一致)
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

        # 2. 数值转换
        for col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # 3. 特征列NaN填充
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        feature_cols_in_df_anomaly = [col for col in df_processed.columns if col not in model_target_cols]
        for col in feature_cols_in_df_anomaly:
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0)

        # 4. 目标列NaN处理 (对于异常检测，如果目标列有NaN，填充为0，因为需要用实际值与预测值比较)
        for tc in model_target_cols:
            if tc in df_processed.columns:
                if df_processed[tc].isnull().any():
                    self.logger.warning(f"警告: 目标列 '{tc}' 在用于异常检测的数据中包含NaN值。这些NaN值将被填充为0。")
                    df_processed[tc] = df_processed[tc].fillna(0)
            else: # 如果目标列完全缺失
                self.logger.error(f"严重警告: 模型训练时定义的目标列 '{tc}' 在提供的异常检测数据中完全缺失。将为此列创建全0数据。")
                df_processed[tc] = 0 # 创建全0列

        # 5. (如果需要) 计算AQI_calculated (与预测/评估一致)
        if 'AQI_calculated' in self.all_feature_columns_for_sequence: # self.all_feature_columns_for_sequence 应已通过_ensure_model_loaded_for_use()加载
            df_processed = calculate_aqi_from_pollutants(df_processed.copy())
            if 'AQI_calculated' in df_processed.columns and df_processed['AQI_calculated'].isnull().any():
                 df_processed['AQI_calculated'] = df_processed['AQI_calculated'].fillna(0)
            elif 'AQI_calculated' not in df_processed.columns:
                 df_processed['AQI_calculated'] = 0

        # 6. 创建周期性时间特征 (与预测/评估一致)
        new_cyclical_features = pd.DataFrame(index=df_processed.index)
        if isinstance(df_processed.index, pd.DatetimeIndex):
            idx = df_processed.index
            # ... (省略重复的周期特征代码)
            new_cyclical_features['hour_sin'] = np.sin(2*np.pi*idx.hour/24.0); new_cyclical_features['hour_cos'] = np.cos(2*np.pi*idx.hour/24.0)
            new_cyclical_features['dayofweek_sin'] = np.sin(2*np.pi*idx.dayofweek/7.0); new_cyclical_features['dayofweek_cos'] = np.cos(2*np.pi*idx.dayofweek/7.0)
            new_cyclical_features['month_sin'] = np.sin(2*np.pi*idx.month/12.0); new_cyclical_features['month_cos'] = np.cos(2*np.pi*idx.month/12.0)
            df_processed = pd.concat([df_processed, new_cyclical_features], axis=1)
        else:
            expected_cyclical_feats = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos']
            for feat_name in expected_cyclical_feats:
                if feat_name in self.all_feature_columns_for_sequence and feat_name not in df_processed.columns:
                    self.logger.warning(f"异常检测数据的索引非DatetimeIndex，但模型期望周期特征 '{feat_name}'。将填充为0。")
                    df_processed[feat_name] = 0

        # 7. 创建滞后特征 (与预测/评估一致)
        lag_features_to_recreate = [f for f in self.all_feature_columns_for_sequence if "_lag_" in f]
        lag_series_list = []
        for lag_col_name in lag_features_to_recreate:
            original_col_name_parts = lag_col_name.split("_lag_")
            if len(original_col_name_parts) == 2:
                original_col = original_col_name_parts[0]; lag_num = int(original_col_name_parts[1])
                if original_col in df_processed.columns:
                    lag_series_list.append(df_processed[original_col].shift(lag_num).rename(lag_col_name))
                else:
                    self.logger.warning(f"警告: 创建滞后特征'{lag_col_name}'所需的原始列'{original_col}'在异常检测数据中未找到。该滞后特征填充为0。")
                    lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))
            else:
                self.logger.warning(f"警告: 滞后特征名 '{lag_col_name}' 格式无法解析。该特征填充为0。")
                lag_series_list.append(pd.Series(0, index=df_processed.index, name=lag_col_name))

        if lag_series_list:
            df_processed = pd.concat([df_processed] + lag_series_list, axis=1)
        # 滞后特征产生的NaN需要填充，因为异常检测可能从序列开头开始
        for col in df_processed.columns: # 遍历所有列，包括新创建的滞后特征
            if df_processed[col].isnull().any(): # 如果有NaN
                df_processed[col] = df_processed[col].ffill().bfill().fillna(0) # 前向、后向、0填充

        # 8. 确保所有模型期望的特征都存在，并按正确的顺序排列 (与预测/评估一致)
        features_dict_anomaly = {}
        for f_col in self.all_feature_columns_for_sequence:
            if f_col in df_processed.columns:
                series_data = pd.to_numeric(df_processed[f_col], errors='coerce').fillna(0)
                if not pd.api.types.is_numeric_dtype(series_data):
                    self.logger.error(f"异常检测预处理中，特征列 '{f_col}' 无法可靠转换为数值类型。")
                    try: features_dict_anomaly[f_col] = series_data.astype(float)
                    except ValueError: raise ValueError(f"无法将特征列 '{f_col}' 转换为浮点数。")
                else: features_dict_anomaly[f_col] = series_data
            else:
                self.logger.warning(f"模型期望的特征列 '{f_col}' 在异常检测数据特征工程后缺失。将创建为全0列。")
                features_dict_anomaly[f_col] = pd.Series(0, index=df_processed.index, name=f_col)

        df_features_for_scaling = pd.DataFrame(features_dict_anomaly, index=df_processed.index)
        df_features_for_scaling = df_features_for_scaling[self.all_feature_columns_for_sequence] # 保证列序

        # 9. 使用已加载的特征缩放器对特征进行缩放
        df_scaled_features_part = self.feature_scaler.transform(df_features_for_scaling)
        df_scaled_features_df = pd.DataFrame(df_scaled_features_part, columns=self.all_feature_columns_for_sequence, index=df_processed.index)

        # 10. 将原始尺度的目标值合并回DataFrame，用于后续create_sequences
        df_final_for_sequences = df_scaled_features_df.copy()
        for tc in model_target_cols: # model_target_cols 已在前面定义
            if tc in df_processed.columns: # df_processed 中的目标列是原始尺度的（或填充为0）
                df_final_for_sequences[tc] = df_processed[tc]
            else: # 这不应该发生
                 self.logger.error(f"严重内部错误: 目标列 '{tc}' 在合并阶段未找到于df_processed。将填充为0。")
                 df_final_for_sequences[tc] = 0 # 理论上不会执行到这里

        # 最后检查是否有NaN，以防万一
        cols_for_sequence_creation = self.all_feature_columns_for_sequence + model_target_cols
        for col in cols_for_sequence_creation:
            if col in df_final_for_sequences.columns and df_final_for_sequences[col].isnull().any():
                self.logger.warning(f"警告: 列 '{col}' 在准备创建序列用于异常检测前仍包含NaN值。将用0填充。")
                df_final_for_sequences[col] = df_final_for_sequences[col].fillna(0)

        if df_final_for_sequences.empty: # 如果处理后数据为空
            raise ValueError("错误: 数据在为异常检测任务预处理完毕后变为空。")

        self.logger.info("异常检测输入数据预处理完成。")
        return df_final_for_sequences

    def detect_anomalies(self, data_path_or_df, threshold_factor=None):
        """
        使用已加载的模型检测输入数据中的异常点。
        异常的判断基于模型对单步预测的重建误差。如果某个时间点的实际值与模型基于其历史数据
        所做的单步预测值之间的差异超过某个阈值，则该点被认为是异常的。
        阈值通常定义为：平均重建误差 + threshold_factor * 重建误差的标准差。
        参数:
            data_path_or_df (str or pd.DataFrame): 包含特征和实际目标值的数据文件路径或DataFrame。
            threshold_factor (float, optional): 用于定义异常阈值的敏感度因子。
                                               如果为None，则使用配置中的 'anomaly_threshold_factor'。
        返回:
            dict: 包含各目标污染物异常检测报告的字典。
                  键为目标污染物名称，值为一个字典，包含：
                  'count': 异常点数量,
                  'threshold_value': 使用的异常阈值,
                  'mean_error': 平均重建误差,
                  'std_dev_error': 重建误差标准差,
                  'relative_indices_in_sequence': 异常点在生成序列中的相对索引,
                  'timestamps_of_anomalies': 异常点的时间戳 (如果可用),
                  'anomaly_details': 包含异常点详细信息（时间、实际值、预测值、误差）的列表。
        """
        self._ensure_model_loaded_for_use() # 确保模型已加载
        current_threshold_factor = threshold_factor if threshold_factor is not None \
                                   else self.config.get('anomaly_threshold_factor', DEFAULT_ANOMALY_THRESHOLD_FACTOR)
        self.logger.info(f"开始执行异常数据检测流程，使用的阈值因子为: {current_threshold_factor} ...")

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

        # 2. 预处理数据 (特征缩放，目标保持原始)
        df_for_sequences = self._preprocess_input_for_anomaly(df_raw_full.copy())

        # 3. 创建序列 (X是缩放特征，y是原始目标)
        # 对于异常检测，我们通常做单步预测 (horizon=1) 来评估重建能力
        model_target_cols = self.model_config.get('target_col_names', DEFAULT_TARGET_COL_NAMES)
        # 检查预处理后的DataFrame是否包含所有必要的目标列（且不全为NaN）
        missing_targets_in_processed_df = [tc for tc in model_target_cols
                                           if tc not in df_for_sequences.columns or df_for_sequences[tc].isnull().all()]
        if missing_targets_in_processed_df:
            raise ValueError(f"错误: 异常检测数据在预处理后，缺少一个或多个有效的实际目标观测列: {missing_targets_in_processed_df}。")

        X_anomaly_np, y_actual_anomaly_np = create_sequences(
            df_for_sequences,
            look_back=self.model_config['look_back'],
            horizon=1, # 关键：异常检测基于单步预测的重建误差
            target_col_names=model_target_cols,
            feature_cols=self.all_feature_columns_for_sequence, # 已加载的特征列
            is_predict=False, logger=self.logger
        )

        if X_anomaly_np.size == 0: # 如果无法创建序列
            self.logger.warning("警告: 无法为异常检测创建有效的输入/输出序列。异常检测中止。")
            return {} # 返回空报告

        # 4. 模型进行单步预测
        self.model.eval() # 评估模式
        all_predictions_scaled_list = [] # 存储缩放后的预测
        anomaly_dataset = TensorDataset(torch.from_numpy(X_anomaly_np).float())
        batch_size_for_anomaly = self.config.get('batch_size', DEFAULT_BATCH_SIZE) # 使用配置的批大小
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=batch_size_for_anomaly, shuffle=False)

        with torch.no_grad():
            for x_batch_tuple in anomaly_loader: # 遍历数据
                x_batch = x_batch_tuple[0].to(DEVICE)
                outputs_scaled_batch = self.model(x_batch) # 模型预测 (输出是多步horizon的，但我们只用第一步)
                all_predictions_scaled_list.append(outputs_scaled_batch.cpu().numpy())

        if not all_predictions_scaled_list:
            self.logger.error("错误: 模型未能对异常检测数据生成任何预测输出。异常检测中止。")
            return {}
        predictions_scaled_full_horizon = np.concatenate(all_predictions_scaled_list, axis=0) # (样本数, horizon, 目标数)
        # 取第一个时间步的预测 (因为我们是基于单步重建误差)
        predictions_scaled_one_step = predictions_scaled_full_horizon[:, 0, :] # (样本数, 目标数)

        # 5. 反向缩放预测结果
        predicted_original_one_step = np.zeros_like(predictions_scaled_one_step)
        for i, col_name in enumerate(model_target_cols):
            scaler = self.target_scalers[col_name]
            pred_col_scaled_reshaped = predictions_scaled_one_step[:, i].reshape(-1, 1)
            predicted_original_one_step[:, i] = scaler.inverse_transform(pred_col_scaled_reshaped).flatten()

        # 6. 获取实际目标值 (也是单步的，因为create_sequences中horizon=1)
        actual_original_one_step = y_actual_anomaly_np[:, 0, :] # (样本数, 目标数)

        # 7. 计算重建误差并检测异常
        errors = np.abs(actual_original_one_step - predicted_original_one_step) # 绝对误差 (样本数, 目标数)
        anomaly_reports = {} # 存储报告
        self.logger.info("\n--- 异常数据检测详细报告 ---")

        # 获取对应y序列的时间戳/索引
        num_sequences_generated = X_anomaly_np.shape[0]
        # df_for_sequences 的索引对应于原始数据（预处理后）
        # y_actual_anomaly_np 的第一个时间点对应于 df_for_sequences 中 look_back 之后的时间点
        if len(df_for_sequences.index) >= self.model_config['look_back'] + num_sequences_generated:
            base_timestamps_for_y = df_for_sequences.index[self.model_config['look_back'] : self.model_config['look_back'] + num_sequences_generated]
        else: # 如果时间戳不够，使用简单索引
            self.logger.warning("警告: 用于异常检测的时间戳序列长度不足。将使用简单范围索引。")
            base_timestamps_for_y = pd.RangeIndex(start=0, stop=num_sequences_generated, step=1)


        for i, col_name in enumerate(model_target_cols): # 对每个目标污染物进行分析
            col_errors = errors[:, i] # 该目标的所有样本的误差
            if len(col_errors) == 0: # 如果没有误差数据
                self.logger.info(f"\n目标污染物: {col_name} - 无有效误差数据可供分析。")
                anomaly_reports[col_name] = {'count': 0, 'threshold': np.nan, 'timestamps': [], 'details': []}
                continue

            mean_error = np.mean(col_errors) # 平均误差
            std_error = np.std(col_errors)   # 误差标准差
            # 定义异常阈值
            threshold = mean_error + current_threshold_factor * std_error if std_error > 1e-9 else mean_error + 1e-9 # 避免std为0

            anomaly_flags_col = col_errors > threshold # 判断哪些点的误差超过阈值
            current_col_anomaly_indices_relative = np.where(anomaly_flags_col)[0] # 获取异常点在序列中的相对索引

            # 获取异常点的时间戳
            anomaly_timestamps_col = [base_timestamps_for_y[k] for k in current_col_anomaly_indices_relative if k < len(base_timestamps_for_y)]

            anomaly_details_list = [] # 存储异常点的详细信息
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

            self.logger.info(f"\n目标污染物: {col_name}")
            self.logger.info(f"  统计信息: 平均绝对误差={mean_error:.4f}, 误差标准差={std_error:.4f}")
            self.logger.info(f"  异常判定阈值 (误差 > {threshold:.4f})")
            if len(anomaly_timestamps_col) > 0:
                self.logger.info(f"  检测到 {len(anomaly_timestamps_col)} 个潜在异常点:")
                for detail in anomaly_details_list[:5]: # 打印前5个异常点的详情
                    self.logger.info(f"    - 时间: {detail['timestamp']}, 实际值: {detail['actual_value']:.2f}, "
                          f"预测值: {detail['predicted_value']:.2f}, 误差: {detail['error']:.2f}")
                if len(anomaly_timestamps_col) > 5: # 如果异常点多于5个
                    self.logger.info(f"    ...等 (共 {len(anomaly_timestamps_col)} 个异常点)")
            else:
                self.logger.info("  在此阈值下，未检测到异常点。")

            # 绘制并保存异常点图
            plot_save_prefix_target = os.path.join(self.artifacts_dir, f"anomaly_detection_report_{col_name}")
            plot_anomalies(
                timestamps_or_indices=base_timestamps_for_y, # X轴使用时间戳或索引
                actual_values=actual_original_one_step[:, i], # Y轴使用实际值
                anomaly_indices=current_col_anomaly_indices_relative, # 标记的异常点索引
                target_name=col_name, save_path_prefix=plot_save_prefix_target,
                title_suffix=f"异常点检测 (阈值因子={current_threshold_factor})"
            )
            self.logger.info(f"  '{col_name}' 的异常点可视化图表已保存至: {plot_save_prefix_target}_anomalies.png")

        self.logger.info("\n--- 异常数据检测流程结束 ---")
        return anomaly_reports

if __name__ == "__main__":
    # 设置全局日志记录和stdout/stderr重定向
    # 这应该在任何其他日志记录器被获取或使用之前完成，并且只执行一次。
    log_directory = os.path.join(MODEL_ARTIFACTS_DIR, LOG_DIR_NAME) # 主日志目录
    main_logger = setup_global_logging_and_redirect(log_directory, MAIN_CONSOLE_LOG_FILE_PREFIX)

    main_logger.info(f"系统当前使用的计算设备: {DEVICE}") # 现在这将进入日志文件和控制台

    # --- 命令行用户交互界面 ---
    # 使用print进行用户交互，这些print也会被捕获到日志中
    print("\n您好！请选择要执行的操作:")
    print(" (1: 训练新模型)")
    print(" (2: 评估现有模型性能)")
    print(" (3: 使用现有模型进行预测)")
    print(" (4: 使用现有模型检测异常数据)")
    action = input("请输入选项 (1-4): ").strip() # 获取用户输入
    main_logger.info(f"用户选择的操作: {action}") # 记录用户选择

    # 根据用户选择执行相应操作
    if action == '1': # 训练新模型
        main_logger.info("\n--- 您已选择：1. 训练新模型 ---")
        # 获取模型工件保存目录，默认为全局定义的 MODEL_ARTIFACTS_DIR
        custom_artifacts_dir = input(f"请输入模型文件及相关组件的保存目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        main_logger.info(f"模型工件目录设置为: {custom_artifacts_dir}")
        # 获取训练数据路径，默认为全局定义的 DEFAULT_FILE_PATH
        train_data_path = input(f"请输入训练数据文件的完整路径 (默认为 '{DEFAULT_FILE_PATH}'): ").strip() or DEFAULT_FILE_PATH
        main_logger.info(f"训练数据路径设置为: {train_data_path}")
        # 询问是否启用IQR异常检测
        enable_iqr_input = input(f"是否在训练预处理中启用IQR异常值检测? (y/n, 默认为 {'y' if DEFAULT_ENABLE_IQR_OUTLIER_DETECTION else 'n'}): ").strip().lower()
        main_logger.info(f"IQR检测用户输入: {enable_iqr_input}")
        iqr_override = None # 初始化IQR覆盖设置为None
        if enable_iqr_input == 'y': iqr_override = True
        elif enable_iqr_input == 'n': iqr_override = False

        if not os.path.exists(train_data_path): # 检查训练数据文件是否存在
            main_logger.error(f"错误: 训练数据文件 '{train_data_path}' 未找到。")
        else:
            # 创建AQISystem实例，使用用户指定的工件目录
            system = AQISystem(artifacts_dir=custom_artifacts_dir)
            main_logger.info(f"开始使用数据 '{train_data_path}' 在目录 '{custom_artifacts_dir}' 中训练新模型...")
            system.train_new_model(train_data_path=train_data_path, enable_iqr_detection_override=iqr_override)
            main_logger.info("新模型训练流程已完成。")

    elif action == '2': # 评估现有模型
        main_logger.info("\n--- 您已选择：2. 评估现有模型性能 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        main_logger.info(f"模型工件目录设置为: {custom_artifacts_dir}")
        test_data_path = input("请输入用于训练模型的数据集路径 (CSV 或 Excel格式): ").strip()
        main_logger.info(f"测试数据路径设置为: {test_data_path}")

        if not os.path.exists(custom_artifacts_dir): # 检查模型目录是否存在
            main_logger.error(f"错误: 模型工件目录 '{custom_artifacts_dir}' 未找到。请先训练模型或提供正确路径。")
        elif not os.path.exists(test_data_path): # 检查测试数据文件是否存在
            main_logger.error(f"错误: 测试数据文件 '{test_data_path}' 未找到。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir)
                system.evaluate_existing_model(test_data_path_or_df=test_data_path)
            except Exception as e:
                main_logger.error(f"评估过程中发生错误: {e}", exc_info=True)


    elif action == '3': # 使用现有模型进行预测
        main_logger.info("\n--- 您已选择：3. 使用现有模型进行预测 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        main_logger.info(f"模型工件目录设置为: {custom_artifacts_dir}")
        input_data_source = input("请输入用于预测的输入数据文件路径 (CSV 或 Excel格式，选择训练该模型的数据集): ").strip()
        main_logger.info(f"预测输入数据路径设置为: {input_data_source}")

        if not os.path.exists(custom_artifacts_dir):
             main_logger.error(f"错误: 模型工件目录 '{custom_artifacts_dir}' 未找到。请先训练模型或提供正确路径。")
        elif not os.path.exists(input_data_source):
            main_logger.error(f"错误: 预测输入数据文件 '{input_data_source}' 未找到。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir)
                main_logger.info(f"开始使用位于 '{custom_artifacts_dir}' 的模型对数据 '{input_data_source}' 进行预测...")
                system.predict_with_existing_model(input_data_path_or_df=input_data_source)
                main_logger.info("预测流程已完成。")
            except Exception as e:
                main_logger.error(f"预测过程中发生错误: {e}", exc_info=True)

    elif action == '4': # 使用现有模型检测异常数据
        main_logger.info("\n--- 您已选择：4. 使用现有模型检测异常数据 ---")
        custom_artifacts_dir = input(f"请输入已训练模型文件所在目录 (默认为 '{MODEL_ARTIFACTS_DIR}'): ").strip() or MODEL_ARTIFACTS_DIR
        main_logger.info(f"模型工件目录设置为: {custom_artifacts_dir}")
        anomaly_data_path = input("请输入用于异常检测的数据文件路径 (CSV 或 Excel格式，模型预测值对应的真实值文件，如“南京_AQI_Data_test.xlsx”): ").strip()
        main_logger.info(f"异常检测数据路径设置为: {anomaly_data_path}")
        # 获取用户自定义的异常检测阈值因子
        threshold_factor_str = input(f"请输入异常检测的阈值敏感度因子 (默认为 {DEFAULT_ANOMALY_THRESHOLD_FACTOR}): ").strip()
        main_logger.info(f"阈值因子用户输入: {threshold_factor_str}")
        custom_threshold_factor = None # 初始化为None
        if threshold_factor_str: # 如果用户有输入
            try:
                custom_threshold_factor = float(threshold_factor_str) # 尝试转换为浮点数
                if custom_threshold_factor <= 0: # 因子必须为正
                     main_logger.warning("警告: 阈值因子必须为正数。将使用默认值。")
                     custom_threshold_factor = None # 重置为None，将使用默认值
            except ValueError: # 如果转换失败
                main_logger.warning("警告: 输入的阈值因子不是有效的数字。将使用默认值。")
                # custom_threshold_factor 保持为 None

        if not os.path.exists(custom_artifacts_dir):
            main_logger.error(f"错误: 模型工件目录 '{custom_artifacts_dir}' 未找到。请先训练模型或提供正确路径。")
        elif not os.path.exists(anomaly_data_path):
            main_logger.error(f"错误: 异常检测数据文件 '{anomaly_data_path}' 未找到。")
        else:
            try:
                system = AQISystem(artifacts_dir=custom_artifacts_dir)
                main_logger.info(f"开始使用位于 '{custom_artifacts_dir}' 的模型对数据 '{anomaly_data_path}' 进行异常检测...")
                anomaly_report = system.detect_anomalies(data_path_or_df=anomaly_data_path, threshold_factor=custom_threshold_factor)
                main_logger.info("异常数据检测流程已完成。")
            except Exception as e:
                main_logger.error(f"异常检测过程中发生错误: {e}", exc_info=True)
    else: # 无效选择
        main_logger.warning("无效的操作选择。请输入 '1' 到 '4'。程序即将退出。")

    main_logger.info("\nAQI预测与异常检测系统运行结束。")
