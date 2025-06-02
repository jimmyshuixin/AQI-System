import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from keras import Input, Model
from keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten, Embedding, Add, Concatenate, Conv1D
from keras_tuner import BayesianOptimization, HyperParameters
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import tempfile # 导入tempfile模块，用于创建临时文件和目录
import shutil # 导入shutil模块，用于高级文件操作（如复制、移动）
import re # 导入re模块，用于正则表达式操作
import math # 导入math模块，提供数学运算函数
import glob # 导入glob模块，用于查找符合特定规则的文件路径名
import pickle # 导入pickle模块，用于Python对象的序列化和反序列化
import logging # 导入logging模块，用于日志记录
from typing import List, Dict, Tuple, Optional, Any # 导入typing模块，用于类型提示

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Matplotlib 中文字符显示配置 ---
def setup_matplotlib_chinese_font():
    """尝试为 Matplotlib 设置一个可用的中文字体。"""
    fonts_to_try = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] 
    for font_name in fonts_to_try:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            logging.info(f"Matplotlib 中文字体已设置为: {font_name}")
            return
        except Exception:
            logging.warning(f"设置 Matplotlib 字体为 {font_name} 失败。")
    logging.warning("未能找到合适的 Matplotlib 中文字体。图表中的中文可能无法正确显示。")

setup_matplotlib_chinese_font()

# --- 常量定义 ---
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
POLLUTANT_BREAKPOINTS = {
    'SO2_24h': [0, 50, 150, 475, 800, 1600, 2100, 2620], 'SO2_1h': [0, 150, 500, 650, 800],
    'NO2_24h': [0, 40, 80, 180, 280, 565, 750, 940], 'NO2_1h': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
    'PM10_24h': [0, 50, 150, 250, 350, 420, 500, 600], 'PM2_5_24h': [0, 35, 75, 115, 150, 250, 350, 500],
    'O3_1h': [0, 160, 200, 300, 400, 800, 1000, 1200], 'O3_8h': [0, 100, 160, 215, 265, 800], 
    'CO_24h': [0, 2, 4, 14, 24, 36, 48, 60], 'CO_1h': [0, 5, 10, 35, 60, 90, 120, 150]
}
TARGET_POLLUTANTS = ['PM2_5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
WEATHER_FEATURES = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure'] 
TIME_FEATURES = ['Hour', 'DayOfWeek', 'DayOfYear', 'Month', 'Year', 'IsWeekend']
AQI_FEATURE = 'AQI'

DEFAULT_OUTPUT_DIR_BASE = "air_quality_prediction_results"
DEFAULT_MODEL_NAME = "model.keras"
DEFAULT_SCALER_NAME = "scaler.pkl" 
DEFAULT_FEATURES_NAME = "feature_columns.pkl" 

DEFAULT_TUNER_PROJECT_NAME_BASE = "air_quality_tuning"
DEFAULT_FORECAST_OUTPUT_DIR_NAME = "future_forecasts"
FEEDBACK_DATA_DIR_NAME = "feedback_data" 
DEFAULT_FUTURE_PREDICTION_STEPS = 72 
DEFAULT_FINETUNE_EPOCHS = 10 

MSG_ERROR_TITLE = "错误"
MSG_INFO_TITLE = "信息"
MSG_WARN_TITLE = "警告"
MSG_DIR_SELECT_DATA_ROOT = "请选择包含所有城市空气质量数据的根文件夹"
MSG_DIR_SELECT_OUTPUT_BASE = "请选择基础输出结果的文件夹"
MSG_FILE_SELECT_PREDICTION_CSV = "请选择之前生成的预测CSV文件"
MSG_DIR_SELECT_TRUTH_FOLDER = "请选择包含未来真实值CSV文件的文件夹" 
MSG_NO_FILE_SELECTED = "未选择文件。"
MSG_NO_DIR_SELECTED = "未选择文件夹。"
MSG_ANALYSIS_COMPLETE = "{} 的分析完成！结果已保存到: {}"
MSG_FINETUNE_COMPLETE = "城市 '{}' 的模型已使用新数据自动微调并保存。"
MSG_EVALUATION_COMPLETE = "预测评估完成。指标如下：\n{}"
MSG_FUTURE_PRED_PROMPT = "请输入要进行未来预测的城市/站点名称。\n训练过的城市包括: {}\n(如果城市名包含空格，请确保输入正确)"
MSG_CITY_NOT_FOUND_FOR_TRAINING = "目标城市 '{}' 的数据列未在CSV文件中找到，或筛选/处理后数据为空。"
MSG_MODEL_NOT_TRAINED_FOR_CITY = "城市 '{}' 的模型尚未训练或加载，无法进行预测。"
MSG_MISSING_TEST_TRUTH = "由于缺少测试集真实值，无法生成 {} 的对比图表。"
MSG_POLLUTANT_TYPE_COL_MISSING = "数据透视所需的污染物类型列 ('type') 未找到。"
MSG_IDENTIFIER_COLS_MISSING = "数据透视所需的标识列 ('date', 'hour') 未找到。"
MSG_FEEDBACK_FILES_MISSING = "请选择预测CSV文件和真实值数据文件夹。" 
MSG_FEEDBACK_DATA_MISMATCH = "预测数据和真实值数据的时间范围或污染物不匹配，无法评估。"
MSG_PROCESSOR_STATE_LOAD_FAIL = "加载模型对应的处理器状态（scaler/特征列）失败。"
MSG_PROCESSOR_NOT_READY_FOR_FINETUNE = "处理器状态未就绪，无法为微调准备数据。请确保模型及其关联状态已正确加载。"

class AQICalculator:
    """根据污染物浓度计算IAQI和AQI。"""
    @staticmethod
    def _calculate_iaqi(cp: float, pollutant_type: str) -> float:
        breakpoint_key_to_try = pollutant_type 
        if pollutant_type == 'PM2.5': breakpoint_key_to_try = 'PM2_5_24h'
        elif pollutant_type == 'PM10': breakpoint_key_to_try = 'PM10_24h'
        elif pollutant_type == 'SO2': breakpoint_key_to_try = 'SO2_24h' 
        elif pollutant_type == 'NO2': breakpoint_key_to_try = 'NO2_24h' 
        elif pollutant_type == 'O3': breakpoint_key_to_try = 'O3_8h'   
        elif pollutant_type == 'CO': breakpoint_key_to_try = 'CO_24h'  
        
        if breakpoint_key_to_try not in POLLUTANT_BREAKPOINTS and pollutant_type in POLLUTANT_BREAKPOINTS:
            breakpoint_key_to_try = pollutant_type
        elif breakpoint_key_to_try not in POLLUTANT_BREAKPOINTS:
            found_key = None
            for key in POLLUTANT_BREAKPOINTS.keys():
                if pollutant_type.upper() in key.upper(): 
                    found_key = key; break
            if found_key: breakpoint_key_to_try = found_key
            else: logging.debug(f"污染物类型 {pollutant_type} 在断点中未找到。IAQI=0。"); return 0.0

        bp = POLLUTANT_BREAKPOINTS[breakpoint_key_to_try]
        i_high, i_low, bp_high, bp_low = 0, 0, 0, 0
        for i in range(len(bp) - 1):
            if bp[i] <= cp < bp[i+1]:
                bp_low, bp_high, i_low, i_high = bp[i], bp[i+1], IAQI_LEVELS[i], IAQI_LEVELS[i+1]; break
        else:
            if cp >= bp[-1]: bp_low, bp_high, i_low, i_high = bp[-2], bp[-1], IAQI_LEVELS[-2], IAQI_LEVELS[-1]
            else: logging.debug(f"污染物 {pollutant_type} 浓度 {cp} 超出定义断点。IAQI=0。"); return 0.0
        if bp_high == bp_low: return float(i_low)
        iaqi = ((i_high - i_low) / (bp_high - bp_low)) * (cp - bp_low) + i_low
        return round(iaqi)

    @staticmethod
    def calculate_aqi_from_df(df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        iaqi_cols = []
        for p_base in TARGET_POLLUTANTS: 
            if p_base in df_copy.columns:
                iaqi_col_name = f"IAQI_{p_base}"
                df_copy[iaqi_col_name] = df_copy[p_base].apply(
                    lambda x: AQICalculator._calculate_iaqi(x, p_base) if pd.notna(x) else np.nan)
                iaqi_cols.append(iaqi_col_name)
            else: logging.debug(f"基础污染物列 '{p_base}' 未在数据中找到，无法计算其IAQI。")
        
        if iaqi_cols: df_copy[AQI_FEATURE] = df_copy[iaqi_cols].max(axis=1)
        else: df_copy[AQI_FEATURE] = np.nan; logging.warning("没有生成任何IAQI列，AQI无法计算。")
        return df_copy

class AirQualityDataProcessor:
    """处理数据加载、预处理和特征工程。"""
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.scaler: Optional[MinMaxScaler] = None 
        self.feature_columns: List[str] = [] 
        self._initialize_scaler_and_features() 

    def _initialize_scaler_and_features(self):
        if self.scaler is None:
            self.scaler = MinMaxScaler()

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns and 'hour' in df.columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df['date']):
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df.dropna(subset=['date'], inplace=True) 
                    df['hour'] = df['hour'].astype(int) 
                    df['_datetime_temp'] = df['date'] + pd.to_timedelta(df['hour'], unit='h')
                    df.index = pd.DatetimeIndex(df['_datetime_temp'])
                    df.drop(columns=['_datetime_temp'], inplace=True, errors='ignore')
                    logging.info("成功从 'date' 和 'hour' 列创建了DatetimeIndex。")
                except Exception as e:
                    logging.error(f"从 'date' 和 'hour' 列创建DatetimeIndex失败: {e}。")
                    raise ValueError("无法建立有效的DatetimeIndex用于时间特征提取。")
            else: raise ValueError("DataFrame索引不是DatetimeIndex，且无'date'/'hour'列。")
        df['Hour']=df.index.hour; df['DayOfWeek']=df.index.dayofweek; df['DayOfYear']=df.index.dayofyear
        df['Month']=df.index.month; df['Year']=df.index.year; df['IsWeekend']=(df.index.dayofweek>=5)
        return df

    def _load_and_pivot_city_data_from_path(self, data_folder_path: str, target_city_name: str) -> Optional[pd.DataFrame]:
        """从文件夹加载所有CSV，筛选目标城市数据，合并后进行透视。"""
        list_of_city_data_long = [] 
        
        csv_files_in_folder = glob.glob(os.path.join(data_folder_path, "**", "china_cities_*.csv"), recursive=True)
        if not csv_files_in_folder: 
            logging.warning(f"在文件夹 {data_folder_path} 中未找到 'china_cities_*.csv' 文件。")
            return None
        logging.info(f"从文件夹 '{data_folder_path}' 找到 {len(csv_files_in_folder)} 个CSV。")

        common_id_vars = ['date', 'hour', 'type']
        for i, f_path in enumerate(csv_files_in_folder):
            logging.info(f"正在处理文件 {i+1}/{len(csv_files_in_folder)}: {os.path.basename(f_path)}") 
            try: df_temp = pd.read_csv(f_path, low_memory=False)
            except Exception as e: logging.error(f"读取 {f_path} 失败: {e}"); continue
            if not all(col in df_temp.columns for col in common_id_vars): logging.warning(f"{f_path} 缺少ID列。跳过。"); continue
            if target_city_name not in df_temp.columns: logging.debug(f"{f_path} 未找到城市列 '{target_city_name}'。"); continue
            cols_to_keep = common_id_vars + [target_city_name]
            city_df_slice = df_temp[cols_to_keep].copy()
            city_df_slice.rename(columns={target_city_name: 'PollutantValue'}, inplace=True)
            list_of_city_data_long.append(city_df_slice)
        
        if not list_of_city_data_long: logging.warning(f"未能从 '{data_folder_path}' 为 '{target_city_name}' 提取数据。"); return None
        combined_long_df = pd.concat(list_of_city_data_long, ignore_index=True)
        logging.info(f"为 '{target_city_name}' 从 '{data_folder_path}' 合并后长格式形状: {combined_long_df.shape}")
        try:
            logging.debug("开始将PollutantValue转换为数值类型...") 
            combined_long_df['PollutantValue'] = pd.to_numeric(combined_long_df['PollutantValue'], errors='coerce')
            logging.debug("PollutantValue转换完成。开始透视操作...")
            df_pivoted = combined_long_df.pivot_table(index=['date', 'hour'], columns='type', values='PollutantValue', aggfunc='mean').reset_index()
            logging.debug("透视操作完成。")
            standardized_cols = {col: str(col).replace('.', '_') for col in df_pivoted.columns}
            df_pivoted.rename(columns=standardized_cols, inplace=True)
            df_pivoted.drop_duplicates(subset=['date', 'hour'], keep='last', inplace=True) 
            return df_pivoted
        except Exception as e: logging.error(f"为 '{target_city_name}' 数据透视失败: {e}"); return None


    def load_and_preprocess_data(self, data_path_or_folder: str, 
                                 target_city_to_filter: str, 
                                 is_for_finetuning: bool = False
                                 ) -> Optional[pd.DataFrame]:
        
        df_pivoted_city_data = self._load_and_pivot_city_data_from_path(
            data_path_or_folder, target_city_to_filter
        )

        if df_pivoted_city_data is None or df_pivoted_city_data.empty: return None
        logging.info(f"为城市 '{target_city_to_filter}' 从 '{data_path_or_folder}' 加载并透视数据后形状: {df_pivoted_city_data.shape}")
        df = df_pivoted_city_data.copy()

        for p in TARGET_POLLUTANTS: 
            if p not in df.columns: df[p] = np.nan
        for p in TARGET_POLLUTANTS:
            if p in df.columns: df[p] = pd.to_numeric(df[p], errors='coerce')
        df.ffill(inplace=True); df.bfill(inplace=True)
        df.dropna(subset=TARGET_POLLUTANTS, how='any', inplace=True)
        if df.empty: logging.error(f"城市 '{target_city_to_filter}' 数据在NaN处理后为空。"); return None

        try: df = self._add_time_features(df)
        except ValueError as ve: logging.error(f"为 '{target_city_to_filter}' 添加时间特征失败: {ve}"); return None
        df = AQICalculator.calculate_aqi_from_df(df)
        
        if is_for_finetuning:
            if not self.feature_columns: 
                logging.error(MSG_PROCESSOR_NOT_READY_FOR_FINETUNE); return None
            current_feature_columns = self.feature_columns[:]
        else: 
            current_feature_columns = TARGET_POLLUTANTS[:]
            for wf in WEATHER_FEATURES: 
                if wf in df.columns: current_feature_columns.append(wf)
            current_feature_columns.extend(TIME_FEATURES)
            if AQI_FEATURE in df.columns: current_feature_columns.append(AQI_FEATURE)
            self.feature_columns = current_feature_columns[:] 
        
        _actual_cols_for_scaling = []
        for col in current_feature_columns:
            if col not in df.columns: df[col] = 0 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                 _actual_cols_for_scaling.append(col)
        if not _actual_cols_for_scaling: logging.error("没有数值特征可缩放。"); return None
        
        if is_for_finetuning:
            if self.scaler is None or not hasattr(self.scaler, 'data_max_'): 
                logging.error("微调时，处理器scaler未正确初始化。"); return None
            cols_to_transform = [col for col in self.feature_columns if col in _actual_cols_for_scaling and col in df.columns] 
            if not cols_to_transform: logging.error("微调数据中没有与原始特征匹配的可转换数值列。"); return None
            df_scaled_values = self.scaler.transform(df[cols_to_transform])
            df_scaled = pd.DataFrame(df_scaled_values, columns=cols_to_transform, index=df.index)
        else: 
            if self.scaler is None: self.scaler = MinMaxScaler() 
            df_scaled_values = self.scaler.fit_transform(df[_actual_cols_for_scaling])
            df_scaled = pd.DataFrame(df_scaled_values, columns=_actual_cols_for_scaling, index=df.index)

        for col in self.feature_columns:
            if col not in df_scaled.columns:
                if col in df.columns: df_scaled[col] = df[col]
                else: df_scaled[col] = 0 
        try: df_scaled = df_scaled[self.feature_columns] 
        except KeyError as e: logging.error(f"最终重排df_scaled列时出错: {e}。"); return None

        logging.info(f"城市 '{target_city_to_filter}' 数据预处理完成，形状: {df_scaled.shape}")
        if df_scaled.empty: logging.error(f"城市 '{target_city_to_filter}' 预处理后DataFrame为空。"); return None
        return df_scaled

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if len(data) < self.sequence_length + 1:
            logging.warning(f"数据长度 ({len(data)}) 小于序列长度+1。无法创建序列。")
            return np.array(X), np.array(y)
        
        if list(data.columns) != self.feature_columns:
            logging.warning(f"Create_sequences的输入数据列与self.feature_columns不匹配。期望: {self.feature_columns}, 实际: {list(data.columns)}")
            try: data = data[self.feature_columns]
            except KeyError as e: logging.error(f"无法按self.feature_columns对齐数据进行序列创建: {e}"); raise

        data_values = data.values
        target_indices_in_features = []
        for tp in TARGET_POLLUTANTS:
            try: target_indices_in_features.append(self.feature_columns.index(tp))
            except ValueError: logging.debug(f"目标污染物 '{tp}' 未在特征列 '{self.feature_columns}' 中找到。")
        
        if not target_indices_in_features: raise ValueError("在特征列中未能找到任何目标污染物。")

        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i:(i + self.sequence_length)])
            y.append(data_values[i + self.sequence_length, target_indices_in_features])
        if not X: return np.array(X), np.array(y)
        return np.array(X), np.array(y)

class AirQualityModel:
    """管理Keras模型。"""
    def __init__(self, sequence_length: int, num_features: int, num_target_pollutants: int):
        self.sequence_length = sequence_length; self.num_features = num_features
        self.num_target_pollutants = num_target_pollutants; self.model: Optional[Model] = None

    def _build_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout_rate)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Conv1D(filters=inputs.shape[-1], kernel_size=1)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn); return x

    def build_model(self, hp: Optional[HyperParameters] = None) -> Model:
        hp_units_1 = hp.Int("units_1", 32, 256, 32) if hp else 64
        hp_dropout_1 = hp.Float("dropout_1", 0.1, 0.5, 0.1) if hp else 0.2
        hp_num_transformer_blocks = hp.Int("num_transformer_blocks", 1, 4, 1) if hp else 2
        hp_head_size = hp.Int("head_size", 32, 128, 32) if hp else 64
        hp_num_heads = hp.Int("num_heads", 2, 8, 2) if hp else 4
        hp_ff_dim_transformer = hp.Int("ff_dim_transformer", 32, 256, 32) if hp else 128
        hp_learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]) if hp else 1e-3
        main_input = Input(shape=(self.sequence_length, self.num_features), name="main_input"); x = main_input
        for _ in range(hp_num_transformer_blocks):
            x = self._build_transformer_block(x, hp_head_size, hp_num_heads, hp_ff_dim_transformer, hp_dropout_1)
        x = Flatten()(x); x = Dense(hp_units_1, activation="relu")(x); x = Dropout(hp_dropout_1)(x)
        output = Dense(self.num_target_pollutants, activation="linear")(x) 
        model = Model(inputs=main_input, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"]); self.model = model; return model

    def tune_model(self, X_train, y_train, X_val, y_val, output_dir, project_name, max_trials=10, epochs=10) -> Model:
        tuner_dir = os.path.join(output_dir, "tuner_results")
        os.makedirs(tuner_dir, exist_ok=True) 
        tuner = BayesianOptimization(self.build_model, "val_loss", max_trials, 1, tuner_dir, project_name, overwrite=True)
        tuner.search_space_summary()
        early_stopping = keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        best_hps = tuner.get_best_hyperparameters(1)[0]; logging.info(f"最佳超参数: {best_hps.values}")
        self.model = tuner.hypermodel.build(best_hps); return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32) -> keras.callbacks.History:
        if not self.model: raise ValueError("模型未构建。")
        early_stopping = keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        return history
    
    def finetune(self, X_new_truth, y_new_truth, epochs=DEFAULT_FINETUNE_EPOCHS, batch_size=32) -> Optional[keras.callbacks.History]:
        if not self.model: logging.error("模型未加载，无法微调。"); return None
        logging.info(f"开始使用新数据微调模型，轮数: {epochs}...")
        history = self.model.fit(X_new_truth, y_new_truth, epochs=epochs, batch_size=batch_size, verbose=1)
        logging.info("模型微调完成。"); return history

    def evaluate(self, X_test, y_test) -> Tuple[Dict[str, float], np.ndarray]:
        if not self.model: raise ValueError("模型未训练或加载。")
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0); y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics = {"loss": loss, "mae": mae, "mape": mape, "r2_score": r2, "rmse": rmse}
        logging.info(f"模型评估指标: {metrics}"); return metrics, y_pred

    def predict_future_steps(self, last_sequence: np.ndarray, num_steps: int, feature_columns: List[str]) -> np.ndarray: 
        if not self.model: raise ValueError("模型未训练或加载。")
        if last_sequence.ndim == 2: last_sequence = np.expand_dims(last_sequence, axis=0)
        if last_sequence.shape[1:] != (self.sequence_length, self.num_features):
            raise ValueError(f"输入序列形状({last_sequence.shape})不匹配模型期望(1, {self.sequence_length}, {self.num_features})")
        future_predictions = []; current_sequence = last_sequence.copy()
        target_indices_in_features = []
        if not feature_columns: 
            logging.error("predict_future_steps 未收到 feature_columns。")
            target_indices_in_features = list(range(self.num_target_pollutants)) 
        else:
            for tp in TARGET_POLLUTANTS:
                try: target_indices_in_features.append(feature_columns.index(tp))
                except ValueError: logging.debug(f"目标污染物 '{tp}' 未在提供的 feature_columns 找到。")
        
        for _ in range(num_steps):
            next_step_pred_scaled = self.model.predict(current_sequence, verbose=0)[0] 
            future_predictions.append(next_step_pred_scaled)
            new_row_scaled = current_sequence[0, -1, :].copy()
            if len(next_step_pred_scaled) == self.num_target_pollutants and self.num_target_pollutants == len(target_indices_in_features):
                 for i, idx in enumerate(target_indices_in_features): new_row_scaled[idx] = next_step_pred_scaled[i]
            else: 
                logging.warning(f"预测输出/目标索引/模型目标数不匹配。更新序列可能不完整。")
                update_count = min(len(next_step_pred_scaled), len(target_indices_in_features))
                for i in range(update_count): new_row_scaled[target_indices_in_features[i]] = next_step_pred_scaled[i]
            current_sequence = np.roll(current_sequence, -1, axis=1); current_sequence[0, -1, :] = new_row_scaled
        return np.array(future_predictions)

    def save(self, file_path: str, scaler: Optional[MinMaxScaler] = None, feature_columns: Optional[List[str]] = None):
        if self.model: 
            self.model.save(file_path)
            logging.info(f"模型已保存到 {file_path}")
            if scaler and feature_columns:
                base, ext = os.path.splitext(file_path)
                scaler_path = base + "_" + DEFAULT_SCALER_NAME
                features_path = base + "_" + DEFAULT_FEATURES_NAME
                try:
                    with open(scaler_path, 'wb') as f: pickle.dump(scaler, f)
                    with open(features_path, 'wb') as f: pickle.dump(feature_columns, f)
                    logging.info(f"Scaler已保存到 {scaler_path}")
                    logging.info(f"特征列已保存到 {features_path}")
                except Exception as e:
                    logging.error(f"保存scaler或特征列失败: {e}")
        else: logging.error("没有模型可供保存。")

    def load(self, file_path: str, processor_to_update: Optional[AirQualityDataProcessor] = None) -> bool:
        try: 
            self.model = load_model(file_path)
            logging.info(f"模型已从 {file_path} 加载。")
            self.sequence_length = self.model.input_shape[1]
            self.num_features = self.model.input_shape[2]
            self.num_target_pollutants = self.model.output_shape[1]

            if processor_to_update:
                base, ext = os.path.splitext(file_path)
                scaler_path = base + "_" + DEFAULT_SCALER_NAME
                features_path = base + "_" + DEFAULT_FEATURES_NAME
                scaler_loaded, features_loaded = False, False
                if os.path.exists(scaler_path):
                    try:
                        with open(scaler_path, 'rb') as f: processor_to_update.scaler = pickle.load(f)
                        logging.info(f"Scaler已从 {scaler_path} 加载并设置到处理器。")
                        scaler_loaded = True
                    except Exception as e: logging.error(f"加载scaler失败: {e}")
                else: logging.warning(f"未找到Scaler文件: {scaler_path}")
                if os.path.exists(features_path):
                    try:
                        with open(features_path, 'rb') as f: processor_to_update.feature_columns = pickle.load(f)
                        logging.info(f"特征列已从 {features_path} 加载并设置到处理器。")
                        features_loaded = True
                    except Exception as e: logging.error(f"加载特征列失败: {e}")
                else: logging.warning(f"未找到特征列文件: {features_path}")
                if not (scaler_loaded and features_loaded):
                    logging.warning(MSG_PROCESSOR_STATE_LOAD_FAIL)
                    return False 
            return True 
        except Exception as e: 
            logging.error(f"从 {file_path} 加载模型时出错: {e}"); 
            return False


class PlottingUtils:
    """用于绘制结果的静态工具函数。"""
    @staticmethod
    def plot_loss(history: keras.callbacks.History, output_dir: str, city_name_suffix: str = ""):
        plt.figure(figsize=(10,6)); plt.plot(history.history['loss'],label='训练损失'); plt.plot(history.history['val_loss'],label='验证损失')
        plt.title(f'{city_name_suffix} 模型损失' if city_name_suffix else '模型损失'); plt.xlabel('轮次'); plt.ylabel('损失')
        plt.legend(); plt.grid(True); plt.savefig(os.path.join(output_dir, f"loss_plot_{city_name_suffix}.png")); plt.close()

    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, pollutant_names, output_dir, city_name_suffix="", num_timesteps=200):
        if y_true.ndim==1: y_true=y_true.reshape(-1,1)
        if y_pred.ndim==1: y_pred=y_pred.reshape(-1,1)
        if y_true.shape[0]==0 or y_pred.shape[0]==0: logging.warning(f"真实/预测值为空，不为 {city_name_suffix} 绘图。"); return
        if y_true.shape[1]!=y_pred.shape[1]: logging.error(f"真实/预测值污染物数量不匹配 ({city_name_suffix})。"); return
        num_pollutants=y_true.shape[1]; timesteps=min(num_timesteps,len(y_true))
        for i in range(num_pollutants):
            p_name = pollutant_names[i] if i < len(pollutant_names) else f"P_{i+1}"
            plt.figure(figsize=(12,6)); plt.plot(y_true[:timesteps,i],label='真实',color='b',marker='.'); plt.plot(y_pred[:timesteps,i],label='预测',color='r',ls='--',marker='x')
            plt.title(f'{city_name_suffix} {p_name} - 真实 vs. 预测 (前{timesteps}步)' if city_name_suffix else f'{p_name} - 真实 vs. 预测 (前{timesteps}步)')
            plt.xlabel('时间步'); plt.ylabel(f'{p_name} 浓度(归一化)'); plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{p_name}_pred_vs_actual_{city_name_suffix}.png")); plt.close()

    @staticmethod
    def plot_feedback_comparison(y_pred_df, y_truth_df, pollutant_names, output_dir, city_name, num_timesteps=200):
        """绘制反馈阶段的预测值与真实值对比图。"""
        common_index = y_pred_df.index.intersection(y_truth_df.index)
        if common_index.empty:
            logging.warning(f"城市 {city_name} 的预测数据和真实反馈数据没有共同的时间戳，无法绘制对比图。")
            return
        
        y_pred_aligned = y_pred_df.loc[common_index]
        y_truth_aligned = y_truth_df.loc[common_index]
        timesteps = min(num_timesteps, len(common_index))

        for pollutant in pollutant_names: 
            pred_col_name = pollutant 
            truth_col_name = pollutant
            if pred_col_name in y_pred_aligned.columns and truth_col_name in y_truth_aligned.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(y_truth_aligned[truth_col_name].values[:timesteps], label='反馈真实值', color='blue', marker='.', linestyle='-')
                plt.plot(y_pred_aligned[pred_col_name].values[:timesteps], label='先前预测值', color='red', linestyle='--', marker='x')
                plt.title(f'{city_name} {pollutant} - 预测反馈对比 (前 {timesteps} 步)')
                plt.xlabel('时间步 (对齐后)'); plt.ylabel(f'{pollutant} 浓度')
                plt.legend(); plt.grid(True)
                plt.savefig(os.path.join(output_dir, f"{city_name}_{pollutant}_feedback_comparison.png"))
                plt.close()
            else:
                logging.warning(f"污染物 '{pollutant}' (尝试列: {pred_col_name}/{truth_col_name}) 未在预测或真实反馈数据中都找到，跳过对比图。")


    @staticmethod
    def plot_future_forecast(predictions_values, pollutant_names, output_dir, city_name):
        if predictions_values.ndim==1: predictions_values=predictions_values.reshape(-1,1)
        if predictions_values.shape[0]==0: logging.warning(f"未来预测值为空，不为 {city_name} 绘图。"); return
        num_pollutants=predictions_values.shape[1]
        for i in range(num_pollutants):
            p_name = pollutant_names[i] if i < len(pollutant_names) else f"P_{i+1}"
            plt.figure(figsize=(12,6)); plt.plot(predictions_values[:,i],label=f'{p_name} 预测',color='g',marker='o')
            plt.title(f'{city_name} - {p_name} 未来 {len(predictions_values)} 步预测'); plt.xlabel('未来时间步'); plt.ylabel(f'{p_name} 浓度预测')
            plt.legend(); plt.grid(True); plt.savefig(os.path.join(output_dir, f"{city_name}_{p_name}_future_forecast.png")); plt.close()
            
    @staticmethod
    def plot_metrics_comparison(initial_metrics, final_metrics, output_dir, city_name_suffix=""):
        metrics_to_plot = ['mae','mape','r2_score','rmse']
        valid_initial = {k:v for k,v in initial_metrics.items() if pd.notna(v)} if initial_metrics else {}
        valid_final = {k:v for k,v in final_metrics.items() if pd.notna(v)} if final_metrics else {}
        labels = [m.upper() for m in metrics_to_plot if m in valid_initial and m in valid_final]
        initial_vals = [valid_initial.get(m,0) for m in metrics_to_plot if m in valid_initial and m in valid_final]
        final_vals = [valid_final.get(m,0) for m in metrics_to_plot if m in valid_initial and m in valid_final]
        if not labels: logging.warning("无共同有效指标可对比。"); return
        x=np.arange(len(labels)); width=0.35; fig,ax=plt.subplots(figsize=(12,7))
        r1=ax.bar(x-width/2,initial_vals,width,label='优化前/初始'); r2=ax.bar(x+width/2,final_vals,width,label='优化后/最终')
        ax.set_ylabel('指标值'); ax.set_title(f'{city_name_suffix} 模型性能指标对比' if city_name_suffix else '模型性能指标对比')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.bar_label(r1,padding=3,fmt='%.3f'); ax.bar_label(r2,padding=3,fmt='%.3f')
        fig.tight_layout(); plt.savefig(os.path.join(output_dir, f"metrics_comparison_{city_name_suffix}.png")); plt.close()

class AirQualityPredictor:
    def __init__(self, sequence_length: int = 24, city_specific_output_dir: str = "city_results"):
        self.sequence_length = sequence_length; self.output_dir = city_specific_output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.processor = AirQualityDataProcessor(sequence_length) 
        self.model_pipeline: Optional[AirQualityModel] = None; self.data_scaled: Optional[pd.DataFrame] = None
        self.X_train,self.y_train,self.X_val,self.y_val,self.X_test,self.y_test = (np.array([]) for _ in range(6))
        self.train_df_len, self.val_df_len = 0,0; self.trained_city_name: Optional[str] = None

    def _prepare_data_for_model(self, data_root_folder_path: str, target_city_name: str, 
                                train_split: float = 0.7, val_split: float = 0.15) -> bool: 
        if not self.processor: raise RuntimeError("数据处理器未初始化。")
        self.trained_city_name = target_city_name
        try:
            self.data_scaled = self.processor.load_and_preprocess_data(
                data_root_folder_path, target_city_name, is_for_finetuning=False 
            )
        except (FileNotFoundError, ValueError) as e: logging.error(f"为城市 '{target_city_name}' 加载或预处理数据失败: {e}"); return False 
        if self.data_scaled is None or self.data_scaled.empty: return False
        n = len(self.data_scaled); min_data_needed = (self.sequence_length + 1) * 3 
        if n < min_data_needed: messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_name}' 数据量过少 ({n}行)，需 {min_data_needed} 行。"); return False
        self.train_df_len = int(n*train_split); self.val_df_len = int(n*val_split); test_df_len = n-self.train_df_len-self.val_df_len
        if self.train_df_len < self.sequence_length+1 or self.val_df_len < self.sequence_length+1 or \
           (test_df_len > 0 and test_df_len < self.sequence_length+1):
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_name}' 数据分割后子集量不足。"); return False
        train_df, val_df, test_df = self.data_scaled.iloc[:self.train_df_len], self.data_scaled.iloc[self.train_df_len:self.train_df_len+self.val_df_len], self.data_scaled.iloc[self.train_df_len+self.val_df_len:]
        self.X_train,self.y_train = self.processor.create_sequences(train_df)
        self.X_val,self.y_val = self.processor.create_sequences(val_df)
        if not test_df.empty and test_df_len >= self.sequence_length+1: self.X_test,self.y_test = self.processor.create_sequences(test_df)
        else: self.X_test,self.y_test = np.array([]),np.array([])
        if self.X_train.size==0 or self.X_val.size==0: messagebox.showerror(MSG_ERROR_TITLE, f"为 '{target_city_name}' 创建序列后训练/验证集为空。"); return False
        num_features,num_targets = self.X_train.shape[2], self.y_train.shape[1]
        if num_targets == 0: messagebox.showerror(MSG_ERROR_TITLE, f"为 '{target_city_name}' 准备数据后无目标污染物。"); return False
        self.model_pipeline = AirQualityModel(self.sequence_length, num_features, num_targets); return True

    def train_evaluate_flow(self, data_root_folder_path: str, target_city_name: str, 
                            tune_model_flag: bool = True, max_trials_tuner: int = 5, epochs_tuner: int = 10,
                            epochs_final: int = 50, batch_size_final: int = 32,
                            model_filename: str = DEFAULT_MODEL_NAME) -> Tuple[Optional[Dict], Optional[Dict], bool]:
        initial_model_metrics, final_model_metrics, optimization_succeeded = None, None, False 
        if not self._prepare_data_for_model(data_root_folder_path, target_city_name): 
            return initial_model_metrics, final_model_metrics, optimization_succeeded
        if not self.model_pipeline: messagebox.showerror(MSG_ERROR_TITLE, "模型流程未初始化。"); return initial_model_metrics,final_model_metrics,optimization_succeeded
        tuner_project = f"{target_city_name}_{DEFAULT_TUNER_PROJECT_NAME_BASE}"
        if tune_model_flag:
            logging.info(f"开始为 '{target_city_name}' 调优...")
            try:
                self.model_pipeline.tune_model(self.X_train,self.y_train,self.X_val,self.y_val,self.output_dir,tuner_project,max_trials_tuner,epochs_tuner)
                optimization_succeeded = True; logging.info(f"城市 '{target_city_name}' 调优完成。")
            except Exception as e: logging.error(f"城市 '{target_city_name}' 调优出错: {e}。使用默认参数。");
            if not self.model_pipeline.model: self.model_pipeline.build_model()
        else: logging.info(f"跳过 '{target_city_name}' 调优。"); self.model_pipeline.build_model()
        if not self.model_pipeline.model: logging.error(f"为 '{target_city_name}' 构建模型失败。"); return initial_model_metrics,final_model_metrics,optimization_succeeded
        logging.info(f"开始为 '{target_city_name}' 训练最终模型...")
        final_history = self.model_pipeline.train(self.X_train,self.y_train,self.X_val,self.y_val,epochs_final,batch_size_final)
        PlottingUtils.plot_loss(final_history, self.output_dir, target_city_name)
        final_model_path = os.path.join(self.output_dir, model_filename)
        self.model_pipeline.save(final_model_path, scaler=self.processor.scaler, feature_columns=self.processor.feature_columns) 
        if self.X_test.size > 0 and self.y_test.size > 0:
            final_model_metrics, y_pred_final = self.model_pipeline.evaluate(self.X_test, self.y_test)
            logging.info(f"城市 '{target_city_name}' 最终评估: {final_model_metrics}")
            PlottingUtils.plot_predictions_vs_actual(self.y_test,y_pred_final,TARGET_POLLUTANTS,self.output_dir,target_city_name)
        else: logging.warning(MSG_MISSING_TEST_TRUTH.format(target_city_name)); final_model_metrics = {"info": f"{target_city_name} 无测试集。"}
        return initial_model_metrics, final_model_metrics, optimization_succeeded

    def predict_future(self, city_name_to_predict: str, model_load_path: str, 
                       data_root_folder_path_for_context: str): 
        num_steps = DEFAULT_FUTURE_PREDICTION_STEPS
        if not self.processor: self.processor = AirQualityDataProcessor(self.sequence_length)
        
        if not self.model_pipeline or not self.model_pipeline.model or self.trained_city_name != city_name_to_predict:
            if os.path.exists(model_load_path):
                logging.info(f"加载 '{city_name_to_predict}' 模型: {model_load_path}...")
                temp_model_pipeline = AirQualityModel(self.sequence_length,1,1) 
                if temp_model_pipeline.load(model_load_path, processor_to_update=self.processor):
                    self.model_pipeline = temp_model_pipeline; self.trained_city_name = city_name_to_predict
                    logging.info(f"模型和处理器状态加载成功。特征数: {self.model_pipeline.num_features}")
                else: messagebox.showerror(MSG_ERROR_TITLE, f"加载模型或其处理器状态失败: {model_load_path}"); return
            else: messagebox.showerror(MSG_ERROR_TITLE, MSG_MODEL_NOT_TRAINED_FOR_CITY.format(city_name_to_predict) + f"路径 {model_load_path} 无效。"); return
        
        if not self.model_pipeline or not self.model_pipeline.model: messagebox.showerror(MSG_ERROR_TITLE, MSG_MODEL_NOT_TRAINED_FOR_CITY.format(city_name_to_predict)); return
        if not self.processor.feature_columns or self.processor.scaler is None or not hasattr(self.processor.scaler, 'data_max_'):
            messagebox.showerror(MSG_ERROR_TITLE, f"预测前处理器状态（scaler/特征列）未正确加载或初始化。"); return

        try:
            city_data_scaled = self.processor.load_and_preprocess_data(
                data_root_folder_path_for_context, city_name_to_predict, 
                is_for_finetuning=False 
            )
        except (ValueError,FileNotFoundError) as e: messagebox.showerror(MSG_ERROR_TITLE, f"为 '{city_name_to_predict}' 准备预测数据出错: {e}"); return
        if city_data_scaled is None or len(city_data_scaled) < self.model_pipeline.sequence_length:
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{city_name_to_predict}' 数据不足以获取最新序列 (需 {self.model_pipeline.sequence_length} 条)。"); return
        
        if len(self.processor.feature_columns) != self.model_pipeline.num_features:
             messagebox.showerror(MSG_ERROR_TITLE, f"数据特征数({len(self.processor.feature_columns)})与模型期望({self.model_pipeline.num_features})不符。"); return
        
        last_sequence_df = city_data_scaled.iloc[-self.model_pipeline.sequence_length:]
        try: last_sequence_values = last_sequence_df[self.processor.feature_columns].values
        except KeyError as e: messagebox.showerror(MSG_ERROR_TITLE, f"提取最终序列时列不匹配: {e}"); return
        
        predictions_scaled = self.model_pipeline.predict_future_steps(last_sequence_values, num_steps, self.processor.feature_columns) 
        
        target_indices_inv, current_pollutants_inv = [], []
        for tp in TARGET_POLLUTANTS:
            if tp in self.processor.feature_columns: 
                target_indices_inv.append(self.processor.feature_columns.index(tp)); current_pollutants_inv.append(tp)
        if not target_indices_inv: messagebox.showerror(MSG_ERROR_TITLE, "无法反归一化：未找到目标污染物索引。"); return
        dummy_preds_full = np.zeros((num_steps, len(self.processor.feature_columns)))
        actual_pollutants_out, actual_preds_inv_subset = current_pollutants_inv[:], None
        if predictions_scaled.shape[1] != len(target_indices_inv):
            logging.warning(f"预测输出污染物数({predictions_scaled.shape[1]})与期望反归一化数({len(target_indices_inv)})不符。")
            min_c = min(predictions_scaled.shape[1], len(target_indices_inv))
            for i in range(min_c): dummy_preds_full[:, target_indices_inv[i]] = predictions_scaled[:, i]
            actual_pollutants_out = [current_pollutants_inv[i] for i in range(min_c)]
            actual_preds_inv_subset = self.processor.scaler.inverse_transform(dummy_preds_full)[:, [target_indices_inv[i] for i in range(min_c)]]
        else:
            for i, idx in enumerate(target_indices_inv): dummy_preds_full[:, idx] = predictions_scaled[:, i]
            actual_preds_inv_subset = self.processor.scaler.inverse_transform(dummy_preds_full)[:, target_indices_inv]
        pred_pollutants_df = pd.DataFrame(actual_preds_inv_subset, columns=actual_pollutants_out)
        pred_df_with_aqi = AQICalculator.calculate_aqi_from_df(pred_pollutants_df.copy())
        if AQI_FEATURE not in pred_df_with_aqi.columns: logging.warning(f"未能计算预测AQI。")
        else: logging.info(f"成功为预测值计算AQI。均值: {pred_df_with_aqi[AQI_FEATURE].mean()}")
        forecast_out_dir = os.path.join(self.output_dir, DEFAULT_FORECAST_OUTPUT_DIR_NAME); os.makedirs(forecast_out_dir, exist_ok=True)
        PlottingUtils.plot_future_forecast(actual_preds_inv_subset, actual_pollutants_out, forecast_out_dir, city_name_to_predict)
        last_ts = city_data_scaled.index[-1]; future_ts = pd.date_range(start=last_ts+pd.Timedelta(hours=1), periods=num_steps, freq='h') 
        csv_p = os.path.join(forecast_out_dir, f"{city_name_to_predict}_future_predictions_with_aqi.csv"); pred_df_with_aqi.to_csv(csv_p)
        logging.info(f"{city_name_to_predict} 未来预测(含AQI)已保存至 {csv_p}"); messagebox.showinfo(MSG_INFO_TITLE, f"{city_name_to_predict} 未来预测(含AQI)已保存至 {csv_p}")

    def evaluate_and_finetune_model(self, city_name: str, pred_csv_path: str, truth_data_path: str, 
                                    epochs_finetune: int = DEFAULT_FINETUNE_EPOCHS):
        if not (pred_csv_path and truth_data_path and city_name):
            messagebox.showerror(MSG_ERROR_TITLE, "请提供城市名称、预测文件和真实值数据路径。"); return
        try:
            pred_df = pd.read_csv(pred_csv_path, index_col=0)
            # 尝试更稳健地解析日期索引
            try:
                pred_df.index = pd.to_datetime(pred_df.index)
            except Exception as e_idx:
                logging.warning(f"自动解析预测文件索引为日期时间失败: {e_idx}. 尝试特定格式...")
                try:
                    pred_df.index = pd.to_datetime(pred_df.index, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                except Exception as e_fmt:
                    logging.error(f"使用特定格式解析预测文件索引仍失败: {e_fmt}")
                    messagebox.showerror(MSG_ERROR_TITLE, f"无法解析预测文件 {pred_csv_path} 的日期索引。")
                    return
            pred_df.dropna(subset=[pred_df.index.name], inplace=True) # 删除无法解析为日期的索引行
            if not isinstance(pred_df.index, pd.DatetimeIndex) or pred_df.index.hasnans:
                messagebox.showerror(MSG_ERROR_TITLE, f"预测文件 {pred_csv_path} 的索引未能成功转换为有效的日期时间格式。")
                return

        except Exception as e: messagebox.showerror(MSG_ERROR_TITLE, f"读取预测文件失败: {e}"); return
        
        model_path_for_city = os.path.join(self.output_dir, f"{city_name}_{DEFAULT_MODEL_NAME}")
        if not os.path.exists(model_path_for_city):
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{city_name}' 模型文件 {model_path_for_city} 未找到，无法微调。"); return
        
        if not self.processor: self.processor = AirQualityDataProcessor(self.sequence_length) 
        
        temp_model_pipeline = AirQualityModel(self.sequence_length, 1, 1) 
        if not temp_model_pipeline.load(model_path_for_city, processor_to_update=self.processor):
            messagebox.showerror(MSG_ERROR_TITLE, f"加载模型或其处理器状态失败: {model_path_for_city}"); return
        self.model_pipeline = temp_model_pipeline
        self.trained_city_name = city_name
        
        if not self.processor.feature_columns or self.processor.scaler is None or not hasattr(self.processor.scaler, 'data_max_'):
             messagebox.showerror(MSG_ERROR_TITLE, f"无法处理反馈数据：为城市 '{city_name}' 加载的处理器状态不完整。"); return
        logging.info(f"成功加载城市 '{city_name}' 的模型和处理器状态用于评估和微调。特征列: {self.processor.feature_columns}")

        new_truth_data_scaled = self.processor.load_and_preprocess_data(
            truth_data_path, city_name, is_for_finetuning=True
        )
        if new_truth_data_scaled is None or new_truth_data_scaled.empty:
            messagebox.showerror(MSG_ERROR_TITLE, f"处理城市 '{city_name}' 的新真实值数据失败或数据为空。"); return

        truth_df_for_comp_pivoted = self.processor._load_and_pivot_city_data_from_path(truth_data_path, city_name) 
        if truth_df_for_comp_pivoted is None or truth_df_for_comp_pivoted.empty:
             messagebox.showerror(MSG_ERROR_TITLE, f"无法加载或透视用于比较的真实值数据从 '{truth_data_path}'。"); return
        try: 
            if 'date' in truth_df_for_comp_pivoted.columns and 'hour' in truth_df_for_comp_pivoted.columns:
                if not pd.api.types.is_datetime64_any_dtype(truth_df_for_comp_pivoted['date']):
                    truth_df_for_comp_pivoted['date'] = pd.to_datetime(truth_df_for_comp_pivoted['date'], errors='coerce')
                truth_df_for_comp_pivoted.dropna(subset=['date'], inplace=True)
                truth_df_for_comp_pivoted['_dt_temp'] = truth_df_for_comp_pivoted['date'] + pd.to_timedelta(truth_df_for_comp_pivoted['hour'].astype(int), unit='h')
                truth_df_for_comp_pivoted.set_index(pd.DatetimeIndex(truth_df_for_comp_pivoted['_dt_temp']), inplace=True)
                truth_df_for_comp_pivoted.drop(columns=['_dt_temp','date','hour'], inplace=True, errors='ignore') # 清理列
        except Exception as e: messagebox.showerror(MSG_ERROR_TITLE, f"处理真实值数据（比较用）时间列失败: {e}"); return

        common_cols = [p for p in TARGET_POLLUTANTS if p in pred_df.columns and p in truth_df_for_comp_pivoted.columns]
        if not common_cols: messagebox.showerror(MSG_ERROR_TITLE, "预测数据和真实数据无共同污染物列。"); return
        merged_df = pd.merge(pred_df[common_cols], truth_df_for_comp_pivoted[common_cols], left_index=True, right_index=True, suffixes=('_pred', '_truth'))
        merged_df.dropna(inplace=True)
        if merged_df.empty: messagebox.showerror(MSG_ERROR_TITLE, MSG_FEEDBACK_DATA_MISMATCH); return

        metrics_summary = "评估指标:\n"; eval_metrics = {}
        for p in common_cols:
            yt, yp = merged_df[f"{p}_truth"], merged_df[f"{p}_pred"]
            if len(yt) > 0:
                mae=mean_absolute_error(yt,yp); rmse=np.sqrt(mean_squared_error(yt,yp)); r2=r2_score(yt,yp)
                eval_metrics[p]={'MAE':mae,'RMSE':rmse,'R2':r2}; metrics_summary+=f"  {p}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}\n"
        messagebox.showinfo(MSG_INFO_TITLE, MSG_EVALUATION_COMPLETE.format(metrics_summary))
        feedback_plot_dir = os.path.join(self.output_dir, FEEDBACK_DATA_DIR_NAME); os.makedirs(feedback_plot_dir, exist_ok=True)
        PlottingUtils.plot_feedback_comparison(merged_df[[c+"_pred" for c in common_cols]].rename(columns=lambda x:x.replace("_pred","")), 
                                               merged_df[[c+"_truth" for c in common_cols]].rename(columns=lambda x:x.replace("_truth","")),
                                               common_cols, feedback_plot_dir, city_name)

        logging.info(f"开始为城市 '{city_name}' 自动微调模型...")
        X_new_truth_seq, y_new_truth_seq = self.processor.create_sequences(new_truth_data_scaled) 
        if X_new_truth_seq.size == 0 or y_new_truth_seq.size == 0:
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{city_name}' 从新真实值数据创建序列失败，无法微调。"); return
        finetune_history = self.model_pipeline.finetune(X_new_truth_seq, y_new_truth_seq, epochs=epochs_finetune)
        if finetune_history:
            self.model_pipeline.save(model_path_for_city, scaler=self.processor.scaler, feature_columns=self.processor.feature_columns) 
            messagebox.showinfo(MSG_INFO_TITLE, MSG_FINETUNE_COMPLETE.format(city_name))
        else: messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{city_name}' 模型微调失败。")


class AirQualityAppGUI:
    def __init__(self, root_window):
        self.root = root_window; self.root.title("空气质量智能分析与预测系统"); self.root.geometry("750x700") 
        self.data_root_folder_path_var = tk.StringVar()
        self.base_output_dir_path_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR_BASE)
        self.model_file_to_load_var = tk.StringVar()
        self.target_city_for_training_var = tk.StringVar(value="北京")
        self.sequence_length_var = tk.IntVar(value=24); self.train_split_var = tk.DoubleVar(value=0.7); self.val_split_var = tk.DoubleVar(value=0.15)
        self.tune_model_var = tk.BooleanVar(value=True); self.max_trials_tuner_var = tk.IntVar(value=5)
        self.epochs_tuner_var = tk.IntVar(value=10); self.epochs_final_var = tk.IntVar(value=20); self.batch_size_final_var = tk.IntVar(value=32)
        self.predict_future_var = tk.BooleanVar(value=False) 
        self.predictor_instance: Optional[AirQualityPredictor] = None; self.trained_cities_list = []
        self.city_for_feedback_var = tk.StringVar()
        self.previous_prediction_file_path_var = tk.StringVar()
        self.future_truth_folder_path_var = tk.StringVar() 
        self.finetune_epochs_var = tk.IntVar(value=DEFAULT_FINETUNE_EPOCHS) 
        self._setup_ui()

    def _select_dir(self, string_var, title): dirname=filedialog.askdirectory(title=title); string_var.set(dirname) if dirname else messagebox.showwarning(MSG_WARN_TITLE,MSG_NO_DIR_SELECTED)
    def _select_file(self, string_var, title, is_model=False, is_csv=False): 
        ft = [("Keras 模型","*.keras *.h5")] if is_model else ([("CSV 文件","*.csv")] if is_csv else [("Excel/CSV","*.xlsx *.xls *.csv"),("所有文件","*.*")])
        fn = filedialog.askopenfilename(title=title,filetypes=ft); string_var.set(fn) if fn else messagebox.showwarning(MSG_WARN_TITLE,MSG_NO_FILE_SELECTED)

    def _setup_ui(self):
        main_frame = tk.Frame(self.root,padx=10,pady=10); main_frame.pack(fill=tk.BOTH,expand=True)
        fp_fr = tk.LabelFrame(main_frame,text="数据与路径设置",padx=10,pady=10); fp_fr.pack(fill=tk.X,pady=5)
        tk.Button(fp_fr,text="选择数据集根文件夹",command=lambda:self._select_dir(self.data_root_folder_path_var,MSG_DIR_SELECT_DATA_ROOT)).grid(row=0,column=0,sticky=tk.W,padx=5,pady=2)
        tk.Entry(fp_fr,textvariable=self.data_root_folder_path_var,width=50).grid(row=0,column=1,padx=5,pady=2,sticky=tk.EW)
        tk.Button(fp_fr,text="选择基础输出文件夹",command=lambda:self._select_dir(self.base_output_dir_path_var,MSG_DIR_SELECT_OUTPUT_BASE)).grid(row=1,column=0,sticky=tk.W,padx=5,pady=2)
        tk.Entry(fp_fr,textvariable=self.base_output_dir_path_var,width=50).grid(row=1,column=1,padx=5,pady=2,sticky=tk.EW)
        fp_fr.columnconfigure(1,weight=1)
        ct_fr = tk.LabelFrame(main_frame,text="训练特定城市设置",padx=10,pady=10); ct_fr.pack(fill=tk.X,pady=5)
        tk.Label(ct_fr,text="目标训练城市/站点名:").grid(row=0,column=0,sticky=tk.W,padx=5,pady=2)
        tk.Entry(ct_fr,textvariable=self.target_city_for_training_var,width=20).grid(row=0,column=1,sticky=tk.W,padx=5,pady=2)
        p_fr = tk.LabelFrame(main_frame,text="通用模型参数",padx=10,pady=10); p_fr.pack(fill=tk.X,pady=5)
        tk.Label(p_fr,text="序列长度:").grid(row=0,column=0,sticky=tk.W,padx=5,pady=2); tk.Entry(p_fr,textvariable=self.sequence_length_var,width=7).grid(row=0,column=1,sticky=tk.W,padx=5,pady=2)
        tk.Label(p_fr,text="训练集比例:").grid(row=0,column=2,sticky=tk.W,padx=(10,0),pady=2); tk.Entry(p_fr,textvariable=self.train_split_var,width=7).grid(row=0,column=3,sticky=tk.W,padx=5,pady=2)
        tk.Label(p_fr,text="验证集比例:").grid(row=0,column=4,sticky=tk.W,padx=(10,0),pady=2); tk.Entry(p_fr,textvariable=self.val_split_var,width=7).grid(row=0,column=5,sticky=tk.W,padx=5,pady=2)
        tu_fr = tk.LabelFrame(main_frame,text="模型调优参数",padx=10,pady=10); tu_fr.pack(fill=tk.X,pady=5)
        tk.Checkbutton(tu_fr,text="进行模型调优",variable=self.tune_model_var).grid(row=0,column=0,columnspan=2,sticky=tk.W,padx=5,pady=2)
        tk.Label(tu_fr,text="Tuner最大尝试:").grid(row=1,column=0,sticky=tk.W,padx=5,pady=2); tk.Entry(tu_fr,textvariable=self.max_trials_tuner_var,width=7).grid(row=1,column=1,sticky=tk.W,padx=5,pady=2)
        tk.Label(tu_fr,text="Tuner训练轮次:").grid(row=1,column=2,sticky=tk.W,padx=(10,0),pady=2); tk.Entry(tu_fr,textvariable=self.epochs_tuner_var,width=7).grid(row=1,column=3,sticky=tk.W,padx=5,pady=2)
        tk.Label(tu_fr,text="最终模型轮次:").grid(row=2,column=0,sticky=tk.W,padx=5,pady=2); tk.Entry(tu_fr,textvariable=self.epochs_final_var,width=7).grid(row=2,column=1,sticky=tk.W,padx=5,pady=2)
        tk.Label(tu_fr,text="最终模型批大小:").grid(row=2,column=2,sticky=tk.W,padx=(10,0),pady=2); tk.Entry(tu_fr,textvariable=self.batch_size_final_var,width=7).grid(row=2,column=3,sticky=tk.W,padx=5,pady=2)
        fup_fr = tk.LabelFrame(main_frame,text="未来预测设置 (固定预测3天)",padx=10,pady=10); fup_fr.pack(fill=tk.X,pady=5) 
        tk.Checkbutton(fup_fr,text="进行未来预测",variable=self.predict_future_var).grid(row=0,column=0,columnspan=2,sticky=tk.W,padx=5,pady=2)
        tk.Button(fup_fr,text="选择城市模型文件进行预测",command=lambda:self._select_file(self.model_file_to_load_var,"选择已训练城市模型",is_model=True)).grid(row=1,column=0,sticky=tk.W,pady=2) 
        tk.Entry(fup_fr,textvariable=self.model_file_to_load_var,width=40).grid(row=1,column=1,columnspan=3,sticky=tk.EW,padx=5,pady=2); fup_fr.columnconfigure(1,weight=1)
        
        feedback_frame = tk.LabelFrame(main_frame, text="反馈真实值与模型自动微调", padx=10, pady=10)
        feedback_frame.pack(fill=tk.X, pady=5)
        tk.Label(feedback_frame, text="目标城市名:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Entry(feedback_frame, textvariable=self.city_for_feedback_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(feedback_frame, text="微调轮数:").grid(row=0, column=2, sticky=tk.W, padx=(10,0), pady=2)
        tk.Entry(feedback_frame, textvariable=self.finetune_epochs_var, width=7).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        tk.Button(feedback_frame, text="选择预测CSV文件", command=lambda: self._select_file(self.previous_prediction_file_path_var, MSG_FILE_SELECT_PREDICTION_CSV, is_csv=True)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Entry(feedback_frame, textvariable=self.previous_prediction_file_path_var, width=40).grid(row=1, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        tk.Button(feedback_frame, text="选择真实值数据文件夹", command=lambda: self._select_dir(self.future_truth_folder_path_var, MSG_DIR_SELECT_TRUTH_FOLDER)).grid(row=2, column=0, sticky=tk.W, padx=5, pady=2) 
        tk.Entry(feedback_frame, textvariable=self.future_truth_folder_path_var, width=40).grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        tk.Button(feedback_frame, text="评估并自动微调模型", command=self.run_evaluation_and_finetuning, width=20, bg="lightyellow").grid(row=3, column=0, columnspan=4, pady=5) 
        feedback_frame.columnconfigure(1, weight=1)

        act_fr = tk.Frame(main_frame,pady=10); act_fr.pack(fill=tk.X)
        tk.Button(act_fr,text="开始训练与评估",command=self.run_training_pipeline,width=15,height=2,bg="lightblue").pack(side=tk.LEFT,padx=10,expand=True)
        tk.Button(act_fr,text="执行未来预测",command=self.run_future_prediction,width=15,height=2,bg="lightgreen").pack(side=tk.LEFT,padx=10,expand=True)
        tk.Button(act_fr,text="退出",command=self.root.quit,width=10,height=2,bg="salmon").pack(side=tk.RIGHT,padx=10,expand=True)

    def run_training_pipeline(self):
        data_root_path = self.data_root_folder_path_var.get(); base_output_dir = self.base_output_dir_path_var.get()
        target_city = self.target_city_for_training_var.get().strip()
        if not (data_root_path and base_output_dir and target_city):
            messagebox.showerror(MSG_ERROR_TITLE, "请填写数据集根文件夹、基础输出文件夹和目标训练城市。"); return
        city_specific_output_dir = os.path.join(base_output_dir, target_city)
        try:
            self.predictor_instance = AirQualityPredictor(self.sequence_length_var.get(), city_specific_output_dir)
            logging.info(f"开始为 '{target_city}' 训练与评估...")
            _, final_metrics, _ = self.predictor_instance.train_evaluate_flow(
                data_root_path, target_city, 
                tune_model_flag=self.tune_model_var.get(), max_trials_tuner=self.max_trials_tuner_var.get(),
                epochs_tuner=self.epochs_tuner_var.get(), epochs_final=self.epochs_final_var.get(),
                batch_size_final=self.batch_size_final_var.get(), model_filename=f"{target_city}_{DEFAULT_MODEL_NAME}")
            if final_metrics and not isinstance(final_metrics.get("info"),str):
                messagebox.showinfo(MSG_INFO_TITLE,MSG_ANALYSIS_COMPLETE.format(target_city,city_specific_output_dir))
                if target_city not in self.trained_cities_list: self.trained_cities_list.append(target_city)
        except (ValueError,FileNotFoundError) as ve: logging.error(f"数据准备/文件错误: {ve}") 
        except Exception as e: logging.exception(f"为 '{target_city}' 训练时严重错误:"); messagebox.showerror(MSG_ERROR_TITLE,f"为 '{target_city}' 训练时严重错误: {e}")

    def run_future_prediction(self):
        if not self.predict_future_var.get(): messagebox.showinfo(MSG_INFO_TITLE,"未勾选进行未来预测。"); return
        base_output_dir = self.base_output_dir_path_var.get(); data_root_path = self.data_root_folder_path_var.get()
        if not (base_output_dir and data_root_path): messagebox.showerror(MSG_ERROR_TITLE,"请设置数据集根文件夹和基础输出文件夹。"); return
        
        trained_prompt = f"{','.join(self.trained_cities_list[:5])}{'...' if len(self.trained_cities_list)>5 else ''}" if self.trained_cities_list else "无"
        city_to_predict = simpledialog.askstring("指定预测城市",MSG_FUTURE_PRED_PROMPT.format(trained_prompt),parent=self.root)
        if not city_to_predict: return
        city_to_predict = city_to_predict.strip()
        
        city_model_dir = os.path.join(base_output_dir,city_to_predict)
        model_path = self.model_file_to_load_var.get() or os.path.join(city_model_dir,f"{city_to_predict}_{DEFAULT_MODEL_NAME}")
        if not os.path.exists(model_path): messagebox.showerror(MSG_ERROR_TITLE,f"模型 {model_path} 未找到。"); return
        try:
            predictor = AirQualityPredictor(self.sequence_length_var.get(), city_model_dir)
            logging.info(f"开始为 '{city_to_predict}' 执行未来预测...")
            predictor.predict_future(city_to_predict,model_path,data_root_path) 
        except (ValueError,FileNotFoundError) as ve: logging.error(f"为 '{city_to_predict}' 准备预测数据/加载模型出错: {ve}")
        except Exception as e: logging.exception(f"为 '{city_to_predict}' 预测时出错:"); messagebox.showerror(MSG_ERROR_TITLE,f"为 '{city_to_predict}' 预测时出错: {e}")

    def run_evaluation_and_finetuning(self): 
        city_name = self.city_for_feedback_var.get().strip()
        pred_csv_path = self.previous_prediction_file_path_var.get()
        truth_folder_path = self.future_truth_folder_path_var.get() 
        base_output_dir = self.base_output_dir_path_var.get()
        epochs_finetune = self.finetune_epochs_var.get() 

        if not (city_name and pred_csv_path and truth_folder_path and base_output_dir):
            messagebox.showerror(MSG_ERROR_TITLE, "请提供城市名称、预测文件、真实值数据文件夹和基础输出目录。")
            return

        city_specific_output_dir = os.path.join(base_output_dir, city_name)
        # os.makedirs(city_specific_output_dir, exist_ok=True) # Predictor __init__ 会创建

        try:
            current_predictor = AirQualityPredictor(
                sequence_length=self.sequence_length_var.get(), 
                city_specific_output_dir=city_specific_output_dir
            )
            
            # evaluate_and_finetune_model 内部会加载模型并尝试恢复processor状态
            current_predictor.evaluate_and_finetune_model(
                city_name, pred_csv_path, truth_folder_path, epochs_finetune
            )
            
        except Exception as e:
            logging.exception(f"评估和自动微调城市 '{city_name}' 模型时发生错误:")
            messagebox.showerror(MSG_ERROR_TITLE, f"评估和自动微调城市 '{city_name}' 模型时发生错误: {e}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AirQualityAppGUI(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"GUI 应用程序启动失败或崩溃: {e}", exc_info=True)
        print(f"严重错误: GUI 应用程序失败。请检查日志。错误: {e}")
