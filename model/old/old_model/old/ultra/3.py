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
    'SO2_24h': [0, 50, 150, 475, 800, 1600, 2100, 2620],
    'SO2_1h': [0, 150, 500, 650, 800],
    'NO2_24h': [0, 40, 80, 180, 280, 565, 750, 940],
    'NO2_1h': [0, 100, 200, 700, 1200, 2340, 3090, 3840],
    'PM10_24h': [0, 50, 150, 250, 350, 420, 500, 600],
    'PM2_5_24h': [0, 35, 75, 115, 150, 250, 350, 500],
    'O3_1h': [0, 160, 200, 300, 400, 800, 1000, 1200], 
    'O3_8h': [0, 100, 160, 215, 265, 800], 
    'CO_24h': [0, 2, 4, 14, 24, 36, 48, 60],
    'CO_1h': [0, 5, 10, 35, 60, 90, 120, 150]
}
TARGET_POLLUTANTS = ['PM2_5', 'PM10', 'SO2', 'NO2', 'O3', 'CO'] 
WEATHER_FEATURES = ['Temperature', 'Humidity', 'WindSpeed', 'Pressure'] 
TIME_FEATURES = ['Hour', 'DayOfWeek', 'DayOfYear', 'Month', 'Year', 'IsWeekend']
AQI_FEATURE = 'AQI'
# SITE_ID_FEATURE = 'SiteID' # 已移除站点ID相关功能

DEFAULT_OUTPUT_DIR_BASE = "air_quality_prediction_results"
DEFAULT_MODEL_NAME = "model.keras"
DEFAULT_TUNER_PROJECT_NAME_BASE = "air_quality_tuning"
DEFAULT_FORECAST_OUTPUT_DIR_NAME = "future_forecasts"

MSG_ERROR_TITLE = "错误"
MSG_INFO_TITLE = "信息"
MSG_WARN_TITLE = "警告"
MSG_DIR_SELECT_DATA_ROOT = "请选择包含所有城市空气质量数据的根文件夹"
# MSG_FILE_SELECT_SITE_INFO = "请选择站点信息CSV/Excel文件 (可选)" # 已移除
MSG_DIR_SELECT_OUTPUT_BASE = "请选择基础输出结果的文件夹"
MSG_NO_FILE_SELECTED = "未选择文件。"
MSG_NO_DIR_SELECTED = "未选择文件夹。"
MSG_ANALYSIS_COMPLETE = "{} 的分析完成！结果已保存到: {}"
MSG_FUTURE_PRED_PROMPT = "请输入要进行未来预测的城市/站点名称。\n训练过的城市包括: {}\n(如果城市名包含空格，请确保输入正确)"
MSG_CITY_NOT_FOUND_FOR_TRAINING = "目标城市 '{}' 的数据列未在CSV文件中找到，或筛选/处理后数据为空。"
# MSG_CITY_NOT_FOUND_IN_SITES = "输入的城市/站点 '{}' 未在站点信息中找到。" # 已移除
MSG_MODEL_NOT_TRAINED_FOR_CITY = "城市 '{}' 的模型尚未训练或加载，无法进行预测。"
MSG_MISSING_TEST_TRUTH = "由于缺少测试集真实值，无法生成 {} 的对比图表。"
MSG_POLLUTANT_TYPE_COL_MISSING = "数据透视所需的污染物类型列 ('type') 未找到。"
MSG_IDENTIFIER_COLS_MISSING = "数据透视所需的标识列 ('date', 'hour') 未找到。"


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
                    found_key = key
                    break
            if found_key: breakpoint_key_to_try = found_key
            else:
                logging.debug(f"污染物类型 {pollutant_type} (尝试键: {breakpoint_key_to_try}) 在断点中未找到。IAQI=0。")
                return 0.0

        bp = POLLUTANT_BREAKPOINTS[breakpoint_key_to_try]
        i_high, i_low, bp_high, bp_low = 0, 0, 0, 0
        for i in range(len(bp) - 1):
            if bp[i] <= cp < bp[i+1]:
                bp_low, bp_high = bp[i], bp[i+1]
                i_low, i_high = IAQI_LEVELS[i], IAQI_LEVELS[i+1]
                break
        else:
            if cp >= bp[-1]:
                bp_low, bp_high = bp[-2], bp[-1]
                i_low, i_high = IAQI_LEVELS[-2], IAQI_LEVELS[-1]
            else:
                 logging.debug(f"污染物 {pollutant_type} 浓度 {cp} 超出定义断点。IAQI=0。")
                 return 0.0
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
            else:
                logging.debug(f"基础污染物列 '{p_base}' 未在数据中找到，无法计算其IAQI。")
        
        if iaqi_cols: df_copy[AQI_FEATURE] = df_copy[iaqi_cols].max(axis=1)
        else:
            df_copy[AQI_FEATURE] = np.nan
            logging.warning("没有生成任何IAQI列，AQI无法计算。")
        return df_copy

class AirQualityDataProcessor:
    """处理数据加载、预处理和特征工程。"""
    def __init__(self, sequence_length: int = 24): # 移除了 site_info_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        # self.site_to_id: Dict[str, int] = {} # 已移除
        # self.id_to_site: Dict[int, str] = {} # 已移除
        # self.num_sites = 0 # 已移除
        self.feature_columns: List[str] = []

        # _load_site_mapping 方法调用已移除

    # _load_site_mapping 方法已移除
    # _add_site_id_feature 方法已移除

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'date' in df.columns and 'hour' in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
                
                df['_datetime_temp'] = df['date'] + pd.to_timedelta(df['hour'].astype(int), unit='h')
                df.index = pd.DatetimeIndex(df['_datetime_temp'])
                df.drop(columns=['_datetime_temp'], inplace=True) 
                logging.info("成功从 'date' 和 'hour' 列创建了DatetimeIndex。")
            except Exception as e:
                logging.error(f"从 'date' 和 'hour' 列创建DatetimeIndex失败: {e}。")
                raise ValueError("无法建立有效的DatetimeIndex用于时间特征提取。")
        elif not isinstance(df.index, pd.DatetimeIndex):
             raise ValueError("DataFrame索引不是DatetimeIndex类型，且未找到'date'和'hour'列来构建它。")

        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['DayOfYear'] = df.index.dayofyear
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        df['IsWeekend'] = df.index.dayofweek >= 5
        return df

    def load_and_preprocess_data(self, data_root_folder_path: str, 
                                 target_city_to_filter: str) -> Optional[pd.DataFrame]: # 移除了 raw_city_column_name_in_csv
        all_csv_files = glob.glob(os.path.join(data_root_folder_path, "**", "china_cities_*.csv"), recursive=True)
        if not all_csv_files:
            logging.error(f"在 {data_root_folder_path} 中未找到任何 'china_cities_*.csv' 文件。")
            messagebox.showerror(MSG_ERROR_TITLE, f"在 {data_root_folder_path} 中未找到CSV文件。")
            return None

        list_of_city_dfs = []
        common_id_vars = ['date', 'hour', 'type']

        for f_path in all_csv_files:
            try:
                df_temp = pd.read_csv(f_path, low_memory=False)
                if not all(col in df_temp.columns for col in common_id_vars):
                    logging.warning(f"文件 {f_path} 缺少ID列。跳过。")
                    continue
                
                if target_city_to_filter in df_temp.columns:
                    cols_to_keep = common_id_vars + [target_city_to_filter]
                    city_df_slice = df_temp[cols_to_keep].copy()
                    city_df_slice.rename(columns={target_city_to_filter: 'PollutantValue'}, inplace=True)
                    list_of_city_dfs.append(city_df_slice)
                else:
                    logging.debug(f"文件 {f_path} 中未找到目标城市列 '{target_city_to_filter}'。")

            except Exception as e:
                logging.warning(f"读取或处理文件 {f_path} 失败: {e}")
                continue
        
        if not list_of_city_dfs:
            logging.error(f"未能从任何文件中提取到城市 '{target_city_to_filter}' 的数据。")
            messagebox.showerror(MSG_ERROR_TITLE, f"未能提取到城市 '{target_city_to_filter}' 的数据。")
            return None

        city_specific_df_long = pd.concat(list_of_city_dfs, ignore_index=True)
        logging.info(f"为城市 '{target_city_to_filter}' 合并数据后形状 (长格式): {city_specific_df_long.shape}")

        if 'type' not in city_specific_df_long.columns or \
           not all(col in city_specific_df_long.columns for col in ['date', 'hour']):
            logging.error(MSG_POLLUTANT_TYPE_COL_MISSING + " 或 " + MSG_IDENTIFIER_COLS_MISSING)
            messagebox.showerror(MSG_ERROR_TITLE, MSG_POLLUTANT_TYPE_COL_MISSING + " 或 " + MSG_IDENTIFIER_COLS_MISSING)
            return None
            
        try:
            city_specific_df_long['PollutantValue'] = pd.to_numeric(city_specific_df_long['PollutantValue'], errors='coerce')
            df_pivoted = city_specific_df_long.pivot_table(
                index=['date', 'hour'], columns='type', values='PollutantValue', aggfunc='mean'
            ).reset_index()
        except Exception as e:
            logging.error(f"数据透视 (pivot) 操作失败: {e}")
            messagebox.showerror(MSG_ERROR_TITLE, f"数据透视操作失败: {e}")
            return None

        logging.info(f"城市 '{target_city_to_filter}' 数据透视后形状: {df_pivoted.shape}")
        df = df_pivoted

        standardized_cols = {}
        for col in df.columns: 
            std_col_name = str(col).replace('.', '_') 
            standardized_cols[col] = std_col_name
        df.rename(columns=standardized_cols, inplace=True)
        
        for p in TARGET_POLLUTANTS: 
            if p not in df.columns:
                logging.warning(f"目标污染物列 '{p}' (标准化后) 未找到。将使用 NaN 填充。")
                df[p] = np.nan
        
        for p in TARGET_POLLUTANTS:
            if p in df.columns: df[p] = pd.to_numeric(df[p], errors='coerce')

        df.ffill(inplace=True); df.bfill(inplace=True)
        df.dropna(subset=TARGET_POLLUTANTS, how='any', inplace=True)
        if df.empty:
            logging.error(f"城市 '{target_city_to_filter}' 数据在NaN处理后为空。")
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_to_filter}' 数据在NaN处理后为空。")
            return None

        try:
            df = self._add_time_features(df)
        except ValueError as ve: 
            logging.error(f"为城市 '{target_city_to_filter}' 添加时间特征时失败: {ve}")
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{target_city_to_filter}' 添加时间特征时失败: {ve}")
            return None

        # _add_site_id_feature 调用已移除
        df = AQICalculator.calculate_aqi_from_df(df)
        
        self.feature_columns = TARGET_POLLUTANTS[:]
        for wf in WEATHER_FEATURES: 
            if wf in df.columns: self.feature_columns.append(wf)
        self.feature_columns.extend(TIME_FEATURES)
        if AQI_FEATURE in df.columns: self.feature_columns.append(AQI_FEATURE)
        # SITE_ID_FEATURE 相关逻辑已移除
        
        _actual_feature_columns_for_scaling = []
        for col in self.feature_columns:
            if col not in df.columns: df[col] = 0 
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                 _actual_feature_columns_for_scaling.append(col)
        
        if not _actual_feature_columns_for_scaling:
            logging.error("没有可用的数值特征进行缩放。"); messagebox.showerror(MSG_ERROR_TITLE, "没有可用的数值特征。"); return None
        
        df_scaled_values = self.scaler.fit_transform(df[_actual_feature_columns_for_scaling])
        df_scaled = pd.DataFrame(df_scaled_values, columns=_actual_feature_columns_for_scaling, index=df.index)
        
        for col in self.feature_columns:
            if col not in df_scaled.columns and col in df.columns: df_scaled[col] = df[col]
        
        logging.info(f"城市 '{target_city_to_filter}' 数据预处理完成，形状: {df_scaled.shape}")
        if df_scaled.empty:
            logging.error(f"城市 '{target_city_to_filter}' 预处理后DataFrame为空。"); messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_to_filter}' 预处理后DataFrame为空。"); return None
            
        self.feature_columns = list(df_scaled.columns)
        return df_scaled

    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        if len(data) < self.sequence_length + 1:
            logging.warning(f"数据长度 ({len(data)}) 小于序列长度+1。无法创建序列。")
            return np.array(X), np.array(y)
        
        data_for_sequence = data
        current_feature_cols_for_sequence = self.feature_columns[:] # 使用当前的 feature_columns

        if list(data.columns) != current_feature_cols_for_sequence:
            logging.info("Create_sequences 输入数据列与预期 feature_columns 不完全匹配或顺序不同。尝试按 feature_columns 重排/选择。")
            try:
                cols_present_in_data = [col for col in current_feature_cols_for_sequence if col in data.columns]
                if len(cols_present_in_data) != len(current_feature_cols_for_sequence):
                    logging.warning(f"并非所有期望的特征列 ({current_feature_cols_for_sequence}) 都存在于数据中 ({data.columns.tolist()})。")
                data_for_sequence = data[cols_present_in_data] 
                current_feature_cols_for_sequence = list(data_for_sequence.columns) # 更新为实际使用的列
            except KeyError as e:
                logging.error(f"创建序列时，数据中缺少必要的特征列: {e}")
                raise ValueError(f"创建序列时，数据中缺少必要的特征列: {e}")

        data_values = data_for_sequence.values
        
        target_indices_in_sequence_features = []
        for tp in TARGET_POLLUTANTS:
            try:
                target_indices_in_sequence_features.append(current_feature_cols_for_sequence.index(tp))
            except ValueError:
                logging.debug(f"目标污染物 '{tp}' 未在用于序列的特征列 '{current_feature_cols_for_sequence}' 中找到。")
        
        if not target_indices_in_sequence_features: 
            raise ValueError("在用于序列的特征列中未能找到任何目标污染物。")

        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i:(i + self.sequence_length)])
            y.append(data_values[i + self.sequence_length, target_indices_in_sequence_features])
        if not X: return np.array(X), np.array(y)
        return np.array(X), np.array(y)

class AirQualityModel:
    """管理Keras模型。"""
    def __init__(self, sequence_length: int, num_features: int, num_target_pollutants: int): # 移除了 num_sites
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_target_pollutants = num_target_pollutants
        # self.num_sites = num_sites # 已移除
        self.model: Optional[Model] = None

    def _build_transformer_block(self, inputs, head_size, num_heads, ff_dim, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout_rate)(inputs, inputs)
        x = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Conv1D(filters=inputs.shape[-1], kernel_size=1)(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
        return x

    def build_model(self, hp: Optional[HyperParameters] = None) -> Model:
        hp_units_1 = hp.Int("units_1", 32, 256, step=32) if hp else 64
        hp_dropout_1 = hp.Float("dropout_1", 0.1, 0.5, step=0.1) if hp else 0.2
        hp_num_transformer_blocks = hp.Int("num_transformer_blocks", 1, 4, step=1) if hp else 2
        hp_head_size = hp.Int("head_size", 32, 128, step=32) if hp else 64
        hp_num_heads = hp.Int("num_heads", 2, 8, step=2) if hp else 4
        hp_ff_dim_transformer = hp.Int("ff_dim_transformer", 32, 256, step=32) if hp else 128
        hp_learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]) if hp else 1e-3
        
        main_input = Input(shape=(self.sequence_length, self.num_features), name="main_input")
        x = main_input
        # Site embedding logic was already commented out or not actively used
        for _ in range(hp_num_transformer_blocks):
            x = self._build_transformer_block(x, hp_head_size, hp_num_heads, hp_ff_dim_transformer, hp_dropout_1)
        x = Flatten()(x)
        x = Dense(hp_units_1, activation="relu")(x)
        x = Dropout(hp_dropout_1)(x)
        output = Dense(self.num_target_pollutants, activation="linear")(x) 
        model = Model(inputs=main_input, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mape"])
        self.model = model
        return model

    def tune_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                   output_dir: str, project_name: str, max_trials: int = 10, epochs: int = 10) -> Model:
        tuner_dir = os.path.join(output_dir, "tuner_results")
        tuner = BayesianOptimization(
            hypermodel=self.build_model, objective="val_loss", max_trials=max_trials,
            executions_per_trial=1, directory=tuner_dir, project_name=project_name, overwrite=True) 
        tuner.search_space_summary()
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        logging.info(f"最佳超参数组合: {best_hps.values}")
        self.model = tuner.hypermodel.build(best_hps)
        return self.model

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        if not self.model: raise ValueError("模型未构建。")
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        if not self.model: raise ValueError("模型尚未训练或加载。")
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred); rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics = {"loss": loss, "mae": mae, "mape": mape, "r2_score": r2, "rmse": rmse}
        logging.info(f"模型评估指标: {metrics}")
        return metrics, y_pred

    def predict_future_steps(self, last_sequence: np.ndarray, num_steps: int) -> np.ndarray:
        if not self.model: raise ValueError("模型尚未训练或加载。")
        if last_sequence.ndim == 2: last_sequence = np.expand_dims(last_sequence, axis=0)
        
        if last_sequence.shape[1] != self.sequence_length or last_sequence.shape[2] != self.num_features:
            raise ValueError(f"输入序列形状({last_sequence.shape})不匹配模型期望(1, {self.sequence_length}, {self.num_features})")
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        target_indices_in_features = []
        if not (AirQualityPredictor.processor and AirQualityPredictor.processor.feature_columns):
            logging.error("Processor或其feature_columns在预测未来时不可用。无法确定目标索引。")
            target_indices_in_features = list(range(self.num_target_pollutants))
        else:
            for tp in TARGET_POLLUTANTS:
                try:
                    target_indices_in_features.append(AirQualityPredictor.processor.feature_columns.index(tp))
                except ValueError:
                    logging.debug(f"目标污染物 '{tp}' 未在 processor.feature_columns 中找到，预测时将跳过更新此列。")

        for _ in range(num_steps):
            next_step_pred_scaled = self.model.predict(current_sequence, verbose=0)[0] 
            future_predictions.append(next_step_pred_scaled)
            
            new_row_scaled = current_sequence[0, -1, :].copy()
            
            if len(next_step_pred_scaled) == self.num_target_pollutants and self.num_target_pollutants == len(target_indices_in_features):
                 for i, target_idx_in_all_features in enumerate(target_indices_in_features):
                    new_row_scaled[target_idx_in_all_features] = next_step_pred_scaled[i]
            else: 
                logging.warning(f"预测输出大小({len(next_step_pred_scaled)})与目标索引数({len(target_indices_in_features)})或模型目标数({self.num_target_pollutants})不匹配。更新序列可能不完整。")
                update_count = min(len(next_step_pred_scaled), len(target_indices_in_features))
                for i in range(update_count):
                    new_row_scaled[target_indices_in_features[i]] = next_step_pred_scaled[i]

            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, :] = new_row_scaled
        return np.array(future_predictions)

    def save(self, file_path: str):
        if self.model: self.model.save(file_path); logging.info(f"模型已保存到 {file_path}")
        else: logging.error("没有模型可供保存。")

    def load(self, file_path: str):
        try: self.model = load_model(file_path); logging.info(f"模型已从 {file_path} 加载。")
        except Exception as e: logging.error(f"从 {file_path} 加载模型时出错: {e}"); raise

class PlottingUtils:
    """用于绘制结果的静态工具函数。"""
    @staticmethod
    def plot_loss(history: keras.callbacks.History, output_dir: str, city_name_suffix: str = ""):
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.title(f'{city_name_suffix} 模型损失曲线' if city_name_suffix else '模型损失曲线')
        plt.xlabel('轮次 (Epoch)'); plt.ylabel('损失值 (Loss)')
        plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"loss_plot_{city_name_suffix}.png"))
        plt.close()

    @staticmethod
    def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, pollutant_names: List[str],
                                   output_dir: str, city_name_suffix: str = "", num_timesteps_to_plot: int = 200):
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        if y_true.shape[0] == 0 or y_pred.shape[0] == 0:
            logging.warning(f"真实值或预测值为空，无法为 {city_name_suffix} 绘制预测对比图。")
            return
        if y_true.shape[1] != y_pred.shape[1]:
            logging.error(f"真实值 ({y_true.shape}) 和预测值 ({y_pred.shape}) 的污染物数量不匹配，无法为 {city_name_suffix} 绘图。")
            return
            
        num_pollutants = y_true.shape[1]
        timesteps = min(num_timesteps_to_plot, len(y_true))
        for i in range(num_pollutants):
            pollutant_name = pollutant_names[i] if i < len(pollutant_names) else f"污染物_{i+1}"
            plt.figure(figsize=(12, 6))
            plt.plot(y_true[:timesteps, i], label='真实值', color='blue', marker='.', linestyle='-')
            plt.plot(y_pred[:timesteps, i], label='预测值', color='red', linestyle='--', marker='x')
            plt.title(f'{city_name_suffix} {pollutant_name} - 真实值 vs. 预测值 (前 {timesteps} 步)' if city_name_suffix else f'{pollutant_name} - 真实值 vs. 预测值 (前 {timesteps} 步)')
            plt.xlabel('时间步'); plt.ylabel(f'{pollutant_name} 浓度 (归一化后)')
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{pollutant_name}_pred_vs_actual_{city_name_suffix}.png"))
            plt.close()

    @staticmethod
    def plot_future_forecast(predictions_values: np.ndarray, pollutant_names: List[str],
                             output_dir: str, city_name: str):
        if predictions_values.ndim == 1: predictions_values = predictions_values.reshape(-1,1)
        if predictions_values.shape[0] == 0:
            logging.warning(f"未来预测值为空，无法为 {city_name} 绘制图表。")
            return
        num_pollutants = predictions_values.shape[1]
        for i in range(num_pollutants):
            pollutant_name = pollutant_names[i] if i < len(pollutant_names) else f"污染物_{i+1}"
            plt.figure(figsize=(12, 6))
            plt.plot(predictions_values[:, i], label=f'{pollutant_name} 预测值', color='green', marker='o', linestyle='-')
            plt.title(f'{city_name} - {pollutant_name} 未来 {len(predictions_values)} 时间步预测')
            plt.xlabel('未来时间步'); plt.ylabel(f'{pollutant_name} 浓度预测值')
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{city_name}_{pollutant_name}_future_forecast.png"))
            plt.close()
            
    @staticmethod
    def plot_metrics_comparison(initial_metrics: Dict[str, float], final_metrics: Dict[str, float], output_dir: str, city_name_suffix: str = ""):
        metrics_to_plot = ['mae', 'mape', 'r2_score', 'rmse']
        valid_initial_metrics = {k:v for k,v in initial_metrics.items() if pd.notna(v)} if initial_metrics else {}
        valid_final_metrics = {k:v for k,v in final_metrics.items() if pd.notna(v)} if final_metrics else {}

        labels = [m.upper() for m in metrics_to_plot if m in valid_initial_metrics and m in valid_final_metrics]
        initial_values = [valid_initial_metrics.get(m, 0) for m in metrics_to_plot if m in valid_initial_metrics and m in valid_final_metrics]
        final_values = [valid_final_metrics.get(m, 0) for m in metrics_to_plot if m in valid_initial_metrics and m in valid_final_metrics]
        
        if not labels: logging.warning("没有共同的有效指标可用于绘制对比图。"); return
        x = np.arange(len(labels)); width = 0.35
        fig, ax = plt.subplots(figsize=(12, 7))
        rects1 = ax.bar(x - width/2, initial_values, width, label='优化前/初始模型')
        rects2 = ax.bar(x + width/2, final_values, width, label='优化后/最终模型')
        ax.set_ylabel('指标值'); ax.set_title(f'{city_name_suffix} 模型性能指标对比' if city_name_suffix else '模型性能指标对比')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend()
        ax.bar_label(rects1, padding=3, fmt='%.3f'); ax.bar_label(rects2, padding=3, fmt='%.3f')
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"metrics_comparison_{city_name_suffix}.png"))
        plt.close()

class AirQualityPredictor:
    """主类，用于协调空气质量预测任务。"""
    processor: Optional[AirQualityDataProcessor] = None 

    def __init__(self, sequence_length: int = 24, 
                 city_specific_output_dir: str = "city_results"): # 移除了 site_info_path
        self.sequence_length = sequence_length
        self.output_dir = city_specific_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        AirQualityPredictor.processor = AirQualityDataProcessor(sequence_length) # 移除了 site_info_path
        self.model_pipeline: Optional[AirQualityModel] = None
        self.data_scaled: Optional[pd.DataFrame] = None
        self.X_train, self.y_train = np.array([]), np.array([])
        self.X_val, self.y_val = np.array([]), np.array([])
        self.X_test, self.y_test = np.array([]), np.array([])
        self.train_df_len, self.val_df_len = 0, 0
        self.trained_city_name: Optional[str] = None

    def _prepare_data_for_model(self, data_root_folder_path: str, 
                                target_city_name: str, 
                                # raw_city_column_name_in_csv: str, # 已移除
                                train_split: float = 0.7, val_split: float = 0.15) -> bool:
        if not AirQualityPredictor.processor: raise RuntimeError("数据处理器未初始化。")
        
        self.trained_city_name = target_city_name
        try:
            self.data_scaled = AirQualityPredictor.processor.load_and_preprocess_data(
                data_root_folder_path, target_city_name # 移除了 raw_city_column_name_in_csv
            )
        except (FileNotFoundError, ValueError) as e: 
            logging.error(f"为城市 '{target_city_name}' 加载或预处理数据失败: {e}")
            return False 

        if self.data_scaled is None or self.data_scaled.empty:
            return False

        n = len(self.data_scaled)
        min_data_needed = (self.sequence_length + 1) * 3 
        if n < min_data_needed:
             messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_name}' 数据量过少 ({n}行)，至少需要 {min_data_needed} 行。")
             return False
        self.train_df_len = int(n * train_split); self.val_df_len = int(n * val_split)
        test_df_len = n - self.train_df_len - self.val_df_len
        if self.train_df_len < self.sequence_length + 1 or \
           self.val_df_len < self.sequence_length + 1 or \
           (test_df_len > 0 and test_df_len < self.sequence_length + 1):
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{target_city_name}' 数据分割后，子集数据量不足以创建序列。")
            return False

        train_df = self.data_scaled.iloc[:self.train_df_len]
        val_df = self.data_scaled.iloc[self.train_df_len : self.train_df_len + self.val_df_len]
        test_df = self.data_scaled.iloc[self.train_df_len + self.val_df_len:]

        self.X_train, self.y_train = AirQualityPredictor.processor.create_sequences(train_df)
        self.X_val, self.y_val = AirQualityPredictor.processor.create_sequences(val_df)
        if not test_df.empty and test_df_len >= self.sequence_length + 1:
            self.X_test, self.y_test = AirQualityPredictor.processor.create_sequences(test_df)
        else: self.X_test, self.y_test = np.array([]), np.array([])

        if self.X_train.size == 0 or self.X_val.size == 0:
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{target_city_name}' 创建序列后训练集或验证集为空。")
            return False
        
        num_features = self.X_train.shape[2]
        num_target_pollutants = self.y_train.shape[1]
        if num_target_pollutants == 0: 
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{target_city_name}' 准备数据后，未能确定任何目标污染物进行预测（y_train 为空或列数为0）。请检查TARGET_POLLUTANTS和数据列。")
            return False

        self.model_pipeline = AirQualityModel(
            self.sequence_length, num_features, num_target_pollutants # 移除了 num_sites
        )
        return True

    def train_evaluate_flow(self, data_root_folder_path: str,
                            target_city_name: str, 
                            # raw_city_column_name_in_csv: str, # 已移除
                            tune_model_flag: bool = True, max_trials_tuner: int = 5, epochs_tuner: int = 10,
                            epochs_final: int = 50, batch_size_final: int = 32,
                            model_filename: str = DEFAULT_MODEL_NAME) -> Tuple[Optional[Dict], Optional[Dict], bool]:
        
        # 移除了 raw_city_column_name_in_csv 参数的传递
        if not self._prepare_data_for_model(data_root_folder_path, target_city_name): 
            return None, None, False
        if not self.model_pipeline: 
             messagebox.showerror(MSG_ERROR_TITLE, "模型处理流程未初始化。"); return None, None, False

        initial_model_metrics, final_model_metrics = None, None
        optimization_succeeded = False
        
        tuner_project_name_for_city = f"{target_city_name}_{DEFAULT_TUNER_PROJECT_NAME_BASE}"

        if tune_model_flag:
            logging.info(f"开始为城市 '{target_city_name}' 进行超参数调优...")
            try:
                self.model_pipeline.tune_model(
                    self.X_train, self.y_train, self.X_val, self.y_val,
                    self.output_dir, tuner_project_name_for_city,
                    max_trials=max_trials_tuner, epochs=epochs_tuner)
                optimization_succeeded = True
                logging.info(f"城市 '{target_city_name}' 调优完成。")
            except Exception as e:
                logging.error(f"城市 '{target_city_name}' 模型调优过程中发生错误: {e}。将使用默认参数。")
                if not self.model_pipeline.model: self.model_pipeline.build_model() 
        else:
            logging.info(f"跳过城市 '{target_city_name}' 的超参数调优，使用默认参数。")
            self.model_pipeline.build_model()

        if not self.model_pipeline.model : 
            logging.error(f"为城市 '{target_city_name}' 构建模型失败。")
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{target_city_name}' 构建模型失败。")
            return None, None, False

        logging.info(f"开始为城市 '{target_city_name}' 训练最终模型...")
        final_history = self.model_pipeline.train(
            self.X_train, self.y_train, self.X_val, self.y_val,
            epochs=epochs_final, batch_size=batch_size_final)
        PlottingUtils.plot_loss(final_history, self.output_dir, target_city_name)
        
        final_model_path = os.path.join(self.output_dir, model_filename)
        self.model_pipeline.save(final_model_path)

        if self.X_test.size > 0 and self.y_test.size > 0:
            final_model_metrics, y_pred_final = self.model_pipeline.evaluate(self.X_test, self.y_test)
            logging.info(f"城市 '{target_city_name}' 最终模型评估: {final_model_metrics}")
            PlottingUtils.plot_predictions_vs_actual(
                self.y_test, y_pred_final, TARGET_POLLUTANTS, self.output_dir, target_city_name)
        else:
            logging.warning(MSG_MISSING_TEST_TRUTH.format(target_city_name) + " (最终模型)")
            final_model_metrics = {"info": f"城市 {target_city_name} 无测试集用于最终评估。"}
        
        return initial_model_metrics, final_model_metrics, optimization_succeeded

    def predict_future(self, city_name_to_predict: str, 
                       model_load_path: str, 
                       data_root_folder_path_for_context: str, 
                       # raw_city_column_name_in_csv_for_context: str, # 已移除
                       num_steps: int = 72):
        
        if not AirQualityPredictor.processor:
            AirQualityPredictor.processor = AirQualityDataProcessor(self.sequence_length) # 移除了 site_info_path
            logging.info("为预测场景初始化了 AirQualityDataProcessor。")

        if not self.model_pipeline or not self.model_pipeline.model or self.trained_city_name != city_name_to_predict:
            if os.path.exists(model_load_path):
                logging.info(f"正在从 {model_load_path} 加载城市 '{city_name_to_predict}' 的模型...")
                try:
                    loaded_keras_model = load_model(model_load_path)
                    _seq_len, _num_features = loaded_keras_model.input_shape[1], loaded_keras_model.input_shape[2]
                    _num_targets = loaded_keras_model.output_shape[1]
                    
                    self.model_pipeline = AirQualityModel(_seq_len, _num_features, _num_targets) # 移除了 num_sites
                    self.model_pipeline.model = loaded_keras_model
                    self.trained_city_name = city_name_to_predict
                    logging.info(f"成功加载模型，输入形状: {loaded_keras_model.input_shape}, 输出形状: {loaded_keras_model.output_shape}")
                except Exception as e:
                    messagebox.showerror(MSG_ERROR_TITLE, f"加载模型 {model_load_path} 失败: {e}")
                    return
            else:
                messagebox.showerror(MSG_ERROR_TITLE, MSG_MODEL_NOT_TRAINED_FOR_CITY.format(city_name_to_predict) + f"路径 {model_load_path} 无效。")
                return
        
        if not self.model_pipeline or not self.model_pipeline.model: 
             messagebox.showerror(MSG_ERROR_TITLE, MSG_MODEL_NOT_TRAINED_FOR_CITY.format(city_name_to_predict))
             return

        try:
            city_data_scaled = AirQualityPredictor.processor.load_and_preprocess_data(
                data_root_folder_path_for_context, 
                city_name_to_predict # 移除了 raw_city_column_name_in_csv_for_context
            )
        except (ValueError, FileNotFoundError) as e:
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{city_name_to_predict}' 准备预测数据时出错: {e}")
            return

        if city_data_scaled is None or len(city_data_scaled) < self.model_pipeline.sequence_length:
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{city_name_to_predict}' 的数据不足以获取最新序列 (需要 {self.model_pipeline.sequence_length} 条)。")
            return

        if len(AirQualityPredictor.processor.feature_columns) != self.model_pipeline.num_features:
            logging.error(f"数据处理器准备的特征数 ({len(AirQualityPredictor.processor.feature_columns)}) 与加载模型的期望特征数 ({self.model_pipeline.num_features}) 不符。")
            messagebox.showerror(MSG_ERROR_TITLE, "数据特征与模型期望不符，无法预测。")
            return

        last_sequence_df = city_data_scaled.iloc[-self.model_pipeline.sequence_length:]
        try:
            last_sequence_values = last_sequence_df[AirQualityPredictor.processor.feature_columns].values
        except KeyError as e:
            logging.error(f"提取最终序列时列错误: {e}。期望列: {AirQualityPredictor.processor.feature_columns}，实际: {last_sequence_df.columns.tolist()}")
            messagebox.showerror(MSG_ERROR_TITLE, f"提取最终序列时列不匹配: {e}")
            return
        
        predictions_scaled = self.model_pipeline.predict_future_steps(last_sequence_values, num_steps)

        target_indices_for_inverse = []
        current_pollutant_names_for_inverse = []
        for tp in TARGET_POLLUTANTS: 
            if tp in AirQualityPredictor.processor.feature_columns:
                target_indices_for_inverse.append(AirQualityPredictor.processor.feature_columns.index(tp))
                current_pollutant_names_for_inverse.append(tp)
        
        if not target_indices_for_inverse:
            logging.error("未能确定目标污染物在特征列中的索引，无法反归一化。")
            messagebox.showerror(MSG_ERROR_TITLE, "无法反归一化预测结果：未找到目标污染物索引。")
            return

        dummy_preds_full = np.zeros((num_steps, len(AirQualityPredictor.processor.feature_columns)))
        
        actual_pollutant_names_for_output = current_pollutant_names_for_inverse[:] 
        actual_predictions_inversed_subset = None

        if predictions_scaled.shape[1] != len(target_indices_for_inverse):
            logging.warning(f"预测输出的污染物数量 ({predictions_scaled.shape[1]}) 与期望反归一化的数量 ({len(target_indices_for_inverse)}) 不符。")
            min_cols = min(predictions_scaled.shape[1], len(target_indices_for_inverse))
            for i in range(min_cols):
                dummy_preds_full[:, target_indices_for_inverse[i]] = predictions_scaled[:, i]
            actual_pollutant_names_for_output = [current_pollutant_names_for_inverse[i] for i in range(min_cols)]
            actual_predictions_inversed_subset = AirQualityPredictor.processor.scaler.inverse_transform(dummy_preds_full)[:, [target_indices_for_inverse[i] for i in range(min_cols)]]
        else:
            for i, target_idx in enumerate(target_indices_for_inverse):
                dummy_preds_full[:, target_idx] = predictions_scaled[:, i]
            actual_predictions_inversed_subset = AirQualityPredictor.processor.scaler.inverse_transform(dummy_preds_full)[:, target_indices_for_inverse]
        
        pred_pollutants_df = pd.DataFrame(actual_predictions_inversed_subset, columns=actual_pollutant_names_for_output)
        pred_df_with_aqi = AQICalculator.calculate_aqi_from_df(pred_pollutants_df.copy()) 
        
        if AQI_FEATURE not in pred_df_with_aqi.columns:
            logging.warning(f"未能计算预测AQI值。输出的CSV将不包含AQI列。")
        else:
            logging.info(f"成功为预测值计算了AQI。均值: {pred_df_with_aqi[AQI_FEATURE].mean()}")

        city_forecast_output_dir = os.path.join(self.output_dir, DEFAULT_FORECAST_OUTPUT_DIR_NAME)
        os.makedirs(city_forecast_output_dir, exist_ok=True)

        PlottingUtils.plot_future_forecast(
            actual_predictions_inversed_subset, actual_pollutant_names_for_output, city_forecast_output_dir, city_name_to_predict
        )
        
        last_timestamp = city_data_scaled.index[-1]
        future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=num_steps, freq='H')
        pred_df_with_aqi.index = future_timestamps
        
        csv_path = os.path.join(city_forecast_output_dir, f"{city_name_to_predict}_future_predictions_with_aqi.csv")
        pred_df_with_aqi.to_csv(csv_path)
        
        logging.info(f"{city_name_to_predict} 的未来预测（含AQI）已生成并保存在 {csv_path}")
        messagebox.showinfo(MSG_INFO_TITLE, f"{city_name_to_predict} 的未来预测（含AQI）已生成并保存在 {csv_path}")

class AirQualityAppGUI:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("空气质量智能分析与预测系统 (城市特定模型)")
        self.root.geometry("700x600") # 调整GUI大小以适应移除的控件

        self.data_root_folder_path_var = tk.StringVar()
        # self.site_info_file_path_var = tk.StringVar() # 已移除
        self.base_output_dir_path_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR_BASE)
        self.model_file_to_load_var = tk.StringVar()

        self.target_city_for_training_var = tk.StringVar(value="北京")
        # self.city_column_in_csv_var = tk.StringVar(value="city") # 已移除

        self.sequence_length_var = tk.IntVar(value=24)
        self.train_split_var = tk.DoubleVar(value=0.7)
        self.val_split_var = tk.DoubleVar(value=0.15)
        
        self.tune_model_var = tk.BooleanVar(value=True)
        self.max_trials_tuner_var = tk.IntVar(value=5)
        self.epochs_tuner_var = tk.IntVar(value=10)
        self.epochs_final_var = tk.IntVar(value=20)
        self.batch_size_final_var = tk.IntVar(value=32)

        self.predict_future_var = tk.BooleanVar(value=False)
        self.num_future_steps_var = tk.IntVar(value=72)
        
        self.predictor_instance: Optional[AirQualityPredictor] = None
        self.trained_cities_list = []

        self._setup_ui()

    def _select_dir(self, string_var: tk.StringVar, title: str):
        dirname = filedialog.askdirectory(title=title)
        if dirname: string_var.set(dirname)
        else: messagebox.showwarning(MSG_WARN_TITLE, MSG_NO_DIR_SELECTED)

    def _select_file(self, string_var: tk.StringVar, title: str, is_model_file: bool = False):
        filetypes = [("Keras 模型", "*.keras *.h5")] if is_model_file else \
                    [("Excel 文件", "*.xlsx *.xls"),("CSV 文件", "*.csv"),  ("所有文件", "*.*")]
        filename = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if filename: string_var.set(filename)
        else: messagebox.showwarning(MSG_WARN_TITLE, MSG_NO_FILE_SELECTED)

    def _setup_ui(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10); main_frame.pack(fill=tk.BOTH, expand=True)

        file_path_frame = tk.LabelFrame(main_frame, text="数据与路径设置", padx=10, pady=10)
        file_path_frame.pack(fill=tk.X, pady=5)
        tk.Button(file_path_frame, text="选择数据集根文件夹", command=lambda: self._select_dir(self.data_root_folder_path_var, MSG_DIR_SELECT_DATA_ROOT)).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Entry(file_path_frame, textvariable=self.data_root_folder_path_var, width=50).grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        # 站点信息文件选择相关UI已移除
        # tk.Button(file_path_frame, text="选择站点信息文件 (可选)", command=lambda: self._select_file(self.site_info_file_path_var, MSG_FILE_SELECT_SITE_INFO)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        # tk.Entry(file_path_frame, textvariable=self.site_info_file_path_var, width=50).grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        tk.Button(file_path_frame, text="选择基础输出文件夹", command=lambda: self._select_dir(self.base_output_dir_path_var, MSG_DIR_SELECT_OUTPUT_BASE)).grid(row=1, column=0, sticky=tk.W, padx=5, pady=2) # 行号调整
        tk.Entry(file_path_frame, textvariable=self.base_output_dir_path_var, width=50).grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW) # 行号调整
        file_path_frame.columnconfigure(1, weight=1)

        city_train_frame = tk.LabelFrame(main_frame, text="训练特定城市设置", padx=10, pady=10)
        city_train_frame.pack(fill=tk.X, pady=5)
        tk.Label(city_train_frame, text="目标训练城市/站点名:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Entry(city_train_frame, textvariable=self.target_city_for_training_var, width=20).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2) # 稍微加宽
        # "CSV中城市列名" 相关UI已移除
        # tk.Label(city_train_frame, text="CSV中城市列名(已弃用):").grid(row=0, column=2, sticky=tk.W, padx=(10,0), pady=2)
        # tk.Entry(city_train_frame, textvariable=self.city_column_in_csv_var, width=15, state=tk.DISABLED).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)

        params_frame = tk.LabelFrame(main_frame, text="通用模型参数", padx=10, pady=10)
        params_frame.pack(fill=tk.X, pady=5)
        tk.Label(params_frame, text="序列长度:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        tk.Entry(params_frame, textvariable=self.sequence_length_var, width=7).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(params_frame, text="训练集比例:").grid(row=0, column=2, sticky=tk.W, padx=(10,0), pady=2)
        tk.Entry(params_frame, textvariable=self.train_split_var, width=7).grid(row=0, column=3, sticky=tk.W, padx=5, pady=2)
        tk.Label(params_frame, text="验证集比例:").grid(row=0, column=4, sticky=tk.W, padx=(10,0), pady=2)
        tk.Entry(params_frame, textvariable=self.val_split_var, width=7).grid(row=0, column=5, sticky=tk.W, padx=5, pady=2)

        tuning_frame = tk.LabelFrame(main_frame, text="模型调优参数", padx=10, pady=10)
        tuning_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(tuning_frame, text="进行模型调优", variable=self.tune_model_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        tk.Label(tuning_frame, text="Tuner最大尝试:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2); tk.Entry(tuning_frame, textvariable=self.max_trials_tuner_var, width=7).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(tuning_frame, text="Tuner训练轮次:").grid(row=1, column=2, sticky=tk.W, padx=(10,0), pady=2); tk.Entry(tuning_frame, textvariable=self.epochs_tuner_var, width=7).grid(row=1, column=3, sticky=tk.W, padx=5, pady=2)
        tk.Label(tuning_frame, text="最终模型轮次:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2); tk.Entry(tuning_frame, textvariable=self.epochs_final_var, width=7).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(tuning_frame, text="最终模型批大小:").grid(row=2, column=2, sticky=tk.W, padx=(10,0), pady=2); tk.Entry(tuning_frame, textvariable=self.batch_size_final_var, width=7).grid(row=2, column=3, sticky=tk.W, padx=5, pady=2)

        future_pred_frame = tk.LabelFrame(main_frame, text="未来预测设置", padx=10, pady=10)
        future_pred_frame.pack(fill=tk.X, pady=5)
        tk.Checkbutton(future_pred_frame, text="进行未来预测", variable=self.predict_future_var).grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        tk.Label(future_pred_frame, text="预测步数:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2); tk.Entry(future_pred_frame, textvariable=self.num_future_steps_var, width=7).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Button(future_pred_frame, text="选择城市模型文件进行预测", command=lambda: self._select_file(self.model_file_to_load_var, "选择已训练的城市模型 (.keras)", is_model_file=True)).grid(row=2, column=0, sticky=tk.W, pady=2)
        tk.Entry(future_pred_frame, textvariable=self.model_file_to_load_var, width=40).grid(row=2, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=2)
        future_pred_frame.columnconfigure(1, weight=1)

        action_frame = tk.Frame(main_frame, pady=10); action_frame.pack(fill=tk.X)
        tk.Button(action_frame, text="开始训练与评估", command=self.run_training_pipeline, width=15, height=2, bg="lightblue").pack(side=tk.LEFT, padx=10, expand=True)
        tk.Button(action_frame, text="执行未来预测", command=self.run_future_prediction, width=15, height=2, bg="lightgreen").pack(side=tk.LEFT, padx=10, expand=True)
        tk.Button(action_frame, text="退出", command=self.root.quit, width=10, height=2, bg="salmon").pack(side=tk.RIGHT, padx=10, expand=True)

    def run_training_pipeline(self):
        data_root_path = self.data_root_folder_path_var.get()
        base_output_dir = self.base_output_dir_path_var.get()
        target_city = self.target_city_for_training_var.get().strip()
        # raw_csv_city_col = self.city_column_in_csv_var.get().strip() # 已移除

        if not data_root_path or not base_output_dir or not target_city:
            messagebox.showerror(MSG_ERROR_TITLE, "请填写所有必要路径和参数：数据集根文件夹、基础输出文件夹、目标训练城市。")
            return
        
        city_specific_output_dir = os.path.join(base_output_dir, target_city)

        try:
            self.predictor_instance = AirQualityPredictor(
                sequence_length=self.sequence_length_var.get(),
                city_specific_output_dir=city_specific_output_dir
                # site_info_path 已移除
            )
            logging.info(f"开始为城市 '{target_city}' 进行训练与评估流程...")
            _, final_metrics, _ = self.predictor_instance.train_evaluate_flow(
                data_root_folder_path=data_root_path,
                target_city_name=target_city,
                # raw_city_column_name_in_csv=raw_csv_city_col, # 已移除
                tune_model_flag=self.tune_model_var.get(),
                max_trials_tuner=self.max_trials_tuner_var.get(),
                epochs_tuner=self.epochs_tuner_var.get(),
                epochs_final=self.epochs_final_var.get(),
                batch_size_final=self.batch_size_final_var.get(),
                model_filename=f"{target_city}_{DEFAULT_MODEL_NAME}"
            )
            if final_metrics and not isinstance(final_metrics.get("info"), str) :
                messagebox.showinfo(MSG_INFO_TITLE, MSG_ANALYSIS_COMPLETE.format(target_city, city_specific_output_dir))
                if target_city not in self.trained_cities_list:
                    self.trained_cities_list.append(target_city)
        except (ValueError, FileNotFoundError) as ve: 
            logging.error(f"数据准备或文件错误: {ve}")
        except Exception as e:
            logging.exception(f"为城市 '{target_city}' 进行的训练流程中发生严重错误:")
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{target_city}' 训练时发生严重错误: {e}")

    def run_future_prediction(self):
        if not self.predict_future_var.get():
            messagebox.showinfo(MSG_INFO_TITLE, "未勾选“进行未来预测”选项。")
            return

        base_output_dir = self.base_output_dir_path_var.get()
        data_root_path = self.data_root_folder_path_var.get()
        # raw_csv_city_col = self.city_column_in_csv_var.get().strip() # 已移除

        if not base_output_dir or not data_root_path : 
            messagebox.showerror(MSG_ERROR_TITLE, "请确保已设置数据集根文件夹和基础输出文件夹，才能进行未来预测。")
            return

        trained_cities_prompt = f"{', '.join(self.trained_cities_list[:5])}{'...' if len(self.trained_cities_list) > 5 else ''}" if self.trained_cities_list else "（当前会话未训练任何城市）"
        city_to_predict = simpledialog.askstring("指定预测城市/站点", 
                                                 MSG_FUTURE_PRED_PROMPT.format(trained_cities_prompt), 
                                                 parent=self.root)
        if not city_to_predict: return
        city_to_predict = city_to_predict.strip()

        city_specific_model_dir = os.path.join(base_output_dir, city_to_predict)
        model_path_for_city = self.model_file_to_load_var.get()
        if not model_path_for_city: 
            model_path_for_city = os.path.join(city_specific_model_dir, f"{city_to_predict}_{DEFAULT_MODEL_NAME}")
        
        if not os.path.exists(model_path_for_city):
            messagebox.showerror(MSG_ERROR_TITLE, f"城市 '{city_to_predict}' 的模型文件 {model_path_for_city} 未找到。请先训练该城市的模型或选择正确的模型文件。")
            return

        try:
            predictor_for_future = AirQualityPredictor(
                sequence_length=self.sequence_length_var.get(),
                city_specific_output_dir=city_specific_model_dir
                # site_info_path 已移除
            )
            
            logging.info(f"开始为城市 '{city_to_predict}' 执行未来预测...")
            predictor_for_future.predict_future(
                city_name_to_predict=city_to_predict,
                model_load_path=model_path_for_city,
                data_root_folder_path_for_context=data_root_path,
                # raw_city_column_name_in_csv_for_context=raw_csv_city_col if raw_csv_city_col else "city", # 已移除
                num_steps=self.num_future_steps_var.get()
            )
        except (ValueError, FileNotFoundError) as ve:
             logging.error(f"为城市 '{city_to_predict}' 准备预测数据或加载模型时出错: {ve}")
        except Exception as e:
            logging.exception(f"为城市 '{city_to_predict}' 进行未来预测时发生错误:")
            messagebox.showerror(MSG_ERROR_TITLE, f"为城市 '{city_to_predict}' 预测时发生错误: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = AirQualityAppGUI(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"GUI 应用程序启动失败或崩溃: {e}", exc_info=True)
        print(f"严重错误: GUI 应用程序失败。请检查日志。错误: {e}")
