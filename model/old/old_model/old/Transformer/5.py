import pandas as pd  # 导入pandas库，用于数据处理和操作
import numpy as np  # 导入numpy库，用于数值计算
from sklearn.preprocessing import MinMaxScaler  # 从sklearn库导入MinMaxScaler类，用于数据归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error  # 导入评价指标函数
from keras import Input, Model  # 从keras库导入Input和Model，用于构建神经网络模型
from keras.layers import LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Flatten, Embedding, Add # 更新导入
from keras_tuner import BayesianOptimization  # 从keras_tuner库导入贝叶斯优化模块
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
import os  # 导入os库，用于文件操作
from keras.models import load_model  # 导入keras中的load_model函数，用于加载已保存的模型
import tkinter as tk # 导入tkinter库，用于GUI
from tkinter import filedialog, messagebox # 从tkinter导入filedialog和messagebox模块
import tempfile # 导入tempfile模块，用于创建临时文件夹
import shutil # 导入shutil模块，用于删除文件夹
from tensorflow import keras # Import keras from tensorflow
import tensorflow as tf # Import tensorflow as tf
import re # 导入re模块，用于正则表达式

# --- Matplotlib Configuration for Chinese Characters ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Default to SimHei
    plt.rcParams['axes.unicode_minus'] = False  # Resolve the minus sign display issue
except Exception:
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # Fallback to Microsoft YaHei
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未能设置中文字体 (SimHei 或 Microsoft YaHei)。图表中的中文可能无法正确显示。")
        print("请确保您的系统已安装 SimHei 或 Microsoft YaHei 字体，或者在代码中指定其他可用的中文字体。")
# --- End Matplotlib Configuration ---

# --- IAQI Calculation Constants (China HJ 633-2012) ---
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
POLLUTANT_BREAKPOINTS = {
    'SO2_24h': [0, 50, 150, 475, 800, 1600, 2100, 2620], # µg/m³
    'NO2_24h': [0, 40, 80, 180, 280, 565, 750, 940],    # µg/m³
    'PM10_24h': [0, 50, 150, 250, 350, 420, 500, 600],  # µg/m³
    'CO_24h': [0, 2, 4, 14, 24, 36, 48, 60],          # mg/m³
    'O3_8h': [0, 100, 160, 215, 265, 800],            # µg/m³
    'PM2.5_24h': [0, 35, 75, 115, 150, 250, 350, 500]   # µg/m³
}
O3_IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500] 
O3_IAQI_LEVELS_FOR_CALC = [0, 50, 100, 150, 200, 300] 

MODEL_TO_IAQI_POLLUTANT_MAP = {
    'SO2_24h': 'SO2_24h', 'NO2_24h': 'NO2_24h', 'PM10_24h': 'PM10_24h',
    'CO_24h': 'CO_24h', 'O3_8h': 'O3_8h', 'PM2.5_24h': 'PM2.5_24h'
}

def calculate_iaqi(c_p, pollutant_key):
    if pollutant_key not in POLLUTANT_BREAKPOINTS: return np.nan
    bp_list = POLLUTANT_BREAKPOINTS[pollutant_key]
    iaqi_levels_for_pollutant = O3_IAQI_LEVELS_FOR_CALC if pollutant_key == 'O3_8h' else IAQI_LEVELS
    if pd.isna(c_p) or c_p < 0: return np.nan
    for i in range(len(bp_list) - 1):
        bp_lo, bp_hi = bp_list[i], bp_list[i+1]
        iaqi_lo, iaqi_hi = iaqi_levels_for_pollutant[i], iaqi_levels_for_pollutant[i+1]
        if bp_lo <= c_p <= bp_hi:
            return round(((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (c_p - bp_lo) + iaqi_lo) if bp_hi != bp_lo else iaqi_lo
    if c_p > bp_list[-1]:
        return O3_IAQI_LEVELS[-1] if pollutant_key == 'O3_8h' and c_p > POLLUTANT_BREAKPOINTS['O3_8h'][-1] else iaqi_levels_for_pollutant[-1]
    return np.nan

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x): # x shape: (batch, seq_len, d_model)
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1) # Use tf.shape(x)[1] for dynamic seq_len
        embedded_positions = self.pos_emb(positions) # shape (seq_len, d_model)
        return x + embedded_positions 

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config

class AQIPredictor:
    def __init__(self, data_path, model_path="trained_model.h5", sequence_length=24, lag_steps=3, rolling_windows=[3, 6]):
        self.data_path = data_path
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows
        
        self.base_feature_names = [
            'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
            'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 
            'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 
            'CO', 'CO_24h'
        ]
        self.target_names = self.base_feature_names + ['AQI']
        if 'AQI' in self.base_feature_names: self.target_names = self.base_feature_names
        else:
            temp_targets = self.base_feature_names[:]
            if 'AQI' not in temp_targets: temp_targets.append('AQI')
            self.target_names = temp_targets
        
        self.feature_names = [] 

        self.scaler_X = MinMaxScaler() 
        self.scaler_y = MinMaxScaler() 
        
        self.date_col_name = 'date' 
        self.hour_col_name = 'hour' 
        self.true_absolute_last_datetime = None
        self.last_sequence_for_future_pred_scaled = None 
        self.initial_history_for_future_features_unscaled = None # Store unscaled history for feature calculation
        
        self._load_and_prepare_data() 
    
    def _add_time_features(self, df_input):
        df = df_input.copy()
        df['hour_sin'] = np.sin(2 * np.pi * df[self.hour_col_name]/23.0) 
        df['hour_cos'] = np.cos(2 * np.pi * df[self.hour_col_name]/23.0)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek/6.0) 
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek/6.0)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear/365.0) 
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear/365.0)
        return df

    def _add_lagged_features(self, df_input, cols_to_lag, lag_steps):
        df_lagged = df_input.copy()
        for col in cols_to_lag:
            if col in df_lagged.columns:
                for lag in range(1, lag_steps + 1):
                    df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
        return df_lagged

    def _add_rolling_features(self, df_input, cols_to_roll, windows):
        df_rolled = df_input.copy()
        for col in cols_to_roll:
            if col in df_rolled.columns:
                for window in windows:
                    df_rolled[f'{col}_roll_mean_{window}h'] = df_rolled[col].rolling(window=window, min_periods=1).mean()
                    df_rolled[f'{col}_roll_std_{window}h'] = df_rolled[col].rolling(window=window, min_periods=1).std()
        return df_rolled

    def _create_sequences(self, X_data, y_data, sequence_length):
        Xs, ys = [], []
        for i in range(len(X_data) - sequence_length): 
            Xs.append(X_data[i:(i + sequence_length)]) 
            ys.append(y_data[i + sequence_length])    
        return np.array(Xs), np.array(ys)

    def _load_and_prepare_data(self, file_path=None, is_external_test_set=False):
        path_to_load = file_path if file_path else self.data_path
        try: aqi_data_original = pd.read_excel(path_to_load)
        except Exception as e: print(f"错误: 读取数据文件 {path_to_load} 失败: {e}"); raise
        if aqi_data_original.empty: raise ValueError(f"错误: 读取的数据文件 {path_to_load} 为空。")

        original_cols = aqi_data_original.columns
        check_cols_original = self.base_feature_names + ['AQI', self.date_col_name, self.hour_col_name]
        missing_original_cols = [col for col in check_cols_original if col not in original_cols]
        if missing_original_cols: raise ValueError(f"数据文件 {path_to_load} 缺少原始列: {missing_original_cols}")

        try:
            aqi_data_original['datetime_str_orig'] = aqi_data_original[self.date_col_name].astype(str) + \
                                                     aqi_data_original[self.hour_col_name].astype(str).str.zfill(2)
            aqi_data_original['datetime'] = pd.to_datetime(aqi_data_original['datetime_str_orig'], format='%Y%m%d%H')
            aqi_data_original.sort_values(by='datetime', inplace=True)
            if not is_external_test_set:
                self.true_absolute_last_datetime = aqi_data_original['datetime'].iloc[-1]
                print(f"信息: 主数据集真实最后时间戳: {self.true_absolute_last_datetime}")
        except Exception as e: print(f"错误: 文件 {path_to_load} 日期转换失败: {e}"); raise

        aqi_data = aqi_data_original.copy()
        
        # Add time features BEFORE imputation of target_names, as time features don't depend on target values
        aqi_data = self._add_time_features(aqi_data)
        
        # Define feature_names list if it's the main data loading
        if not is_external_test_set and not self.feature_names: # Initialize self.feature_names only once
            self.feature_names = self.base_feature_names + \
                                 ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                                  'day_of_year_sin', 'day_of_year_cos']
            # Add lagged feature names
            for target_col in self.target_names:
                for lag in range(1, self.lag_steps + 1):
                    self.feature_names.append(f'{target_col}_lag{lag}')
            # Add rolling feature names
            cols_for_rolling = ['PM2.5', 'PM10', 'O3', 'AQI']
            for col_roll in cols_for_rolling:
                for window in self.rolling_windows:
                    self.feature_names.append(f'{col_roll}_roll_mean_{window}h')
                    self.feature_names.append(f'{col_roll}_roll_std_{window}h')
            self.feature_names = list(dict.fromkeys(self.feature_names)) # Ensure unique
            print(f"信息: 最终使用的输入特征 ({len(self.feature_names)}): {self.feature_names}")


        cols_to_impute_initial = self.target_names 
        nan_counts_before_initial = aqi_data[cols_to_impute_initial].isnull().sum()
        if nan_counts_before_initial.sum() > 0:
            print(f"信息: 文件 {path_to_load} - 初始插补前NaN数:\n{nan_counts_before_initial[nan_counts_before_initial > 0]}")
            for col in cols_to_impute_initial:
                if aqi_data[col].isnull().any(): aqi_data[col] = aqi_data[col].ffill().bfill()
        
        # Add lagged and rolling features AFTER initial imputation of base targets
        aqi_data = self._add_lagged_features(aqi_data, self.target_names, self.lag_steps)
        cols_for_rolling = ['PM2.5', 'PM10', 'O3', 'AQI'] 
        aqi_data = self._add_rolling_features(aqi_data, [c for c in cols_for_rolling if c in aqi_data.columns], self.rolling_windows)
        
        cols_for_final_check = self.feature_names + self.target_names
        initial_rows_before_final_drop = len(aqi_data)
        aqi_data.dropna(subset=cols_for_final_check, inplace=True) 
        if initial_rows_before_final_drop > len(aqi_data):
            print(f"信息: 文件 {path_to_load} - 移除了 {initial_rows_before_final_drop - len(aqi_data)} 行，因滞后/滚动特征产生NaN。")

        if len(aqi_data) < self.sequence_length + 1: 
            raise ValueError(f"数据文件 {path_to_load} 在特征工程和NaN处理后数据过少 (少于 {self.sequence_length + 1} 行)，无法创建序列。")
        
        if not is_external_test_set:
            self.last_datetime_for_autoregression_start = aqi_data['datetime'].iloc[-1]
            print(f"信息: 主数据集中用于启动自回归预测的最后一个序列的结束时间点为: {self.last_datetime_for_autoregression_start}")
            # Save history for future prediction feature calculation
            buffer_len = max(self.lag_steps, max(self.rolling_windows))
            history_needed = self.sequence_length + buffer_len
            self.initial_history_df_for_future_features = aqi_data[['datetime', self.hour_col_name] + self.target_names].iloc[-history_needed:].copy()


        X_df_values = aqi_data[self.feature_names].values 
        y_df_values = aqi_data[self.target_names].values
        datetime_values = aqi_data['datetime'].values

        if not is_external_test_set:
            X_scaled = self.scaler_X.fit_transform(X_df_values)
            y_scaled = self.scaler_y.fit_transform(y_df_values)
            self.last_sequence_for_future_pred_scaled = X_scaled[-self.sequence_length:]
            
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.sequence_length)
            datetime_for_y_sequences = datetime_values[self.sequence_length:] 
            if X_sequences.shape[0] == 0: raise ValueError("创建序列后没有足够的样本。")
            train_size = int(X_sequences.shape[0] * 0.8)
            if train_size == 0 or (X_sequences.shape[0] - train_size) == 0: raise ValueError(f"序列数据量太小。")
            self.X_train, self.X_test = X_sequences[:train_size], X_sequences[train_size:]
            self.y_train, self.y_test = y_sequences[:train_size], y_sequences[train_size:]
            self.test_dates = pd.Series(datetime_for_y_sequences[train_size:])
            return None, None, None 
        else: 
            X_ext_scaled = self.scaler_X.transform(X_df_values)
            y_ext_scaled = self.scaler_y.transform(y_df_values) 
            X_ext_sequences, y_ext_sequences = self._create_sequences(X_ext_scaled, y_ext_scaled, self.sequence_length)
            datetime_for_y_ext_sequences = datetime_values[self.sequence_length:]
            if X_ext_sequences.shape[0] == 0: print(f"警告: 外部测试文件 {path_to_load} 创建序列后无样本。"); return None, None, None
            return X_ext_sequences, y_ext_sequences, pd.Series(datetime_for_y_ext_sequences)

    def _transformer_encoder_block(self, inputs, d_model, num_heads, ff_dim, dropout_rate):
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate)(inputs, inputs, inputs) 
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output) 
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2

    def build_model(self, d_model=64, num_transformer_blocks=1, num_heads=4, ff_dim=128, dropout_rate=0.1, learning_rate=0.001):
        inputs = Input(shape=(self.sequence_length, len(self.feature_names))) 
        x = Dense(d_model)(inputs) 
        x = PositionalEncoding(self.sequence_length, d_model)(x) 
        x = Dropout(dropout_rate)(x) 
        for _ in range(num_transformer_blocks):
            x = self._transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
        x = x[:, -1, :] 
        outputs = Dense(len(self.target_names))(x) 
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) 
        return model

    def train_and_evaluate(self, model, epochs=50, X_train_override=None, y_train_override=None, X_test_override=None, y_test_override=None):
        X_tr = X_train_override if X_train_override is not None else self.X_train
        y_tr = y_train_override if y_train_override is not None else self.y_train
        X_te = X_test_override if X_test_override is not None else self.X_test
        y_te_scaled = y_test_override if y_test_override is not None else self.y_test 
        print(f"开始在 {epochs} 个周期内训练模型...")
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1)
        model.fit(X_tr, y_tr, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping, reduce_lr])
        print("模型训练完成。正在进行预测...")
        predictions_scaled = model.predict(X_te) 
        if np.isnan(predictions_scaled).any():
            print(f"警告: 模型预测结果 (缩放后) 包含 {np.sum(np.isnan(predictions_scaled))} 个NaN值。")
            return np.full_like(y_te_scaled, np.nan), self.scaler_y.inverse_transform(y_te_scaled)
        predictions_inversed = self.scaler_y.inverse_transform(predictions_scaled)
        y_test_inversed = self.scaler_y.inverse_transform(y_te_scaled) 
        if np.isnan(predictions_inversed).any(): print(f"警告: 模型预测结果 (逆缩放后) 包含NaN值。")
        if np.isnan(y_test_inversed).any(): print(f"警告: 测试集真实值 (逆缩放后) 包含NaN值。")
        return predictions_inversed, y_test_inversed

    def save_model(self, model):
        try:
            save_path = self.model_path
            if not save_path.endswith(".keras"): save_path = os.path.splitext(save_path)[0] + ".keras"
            model.save(save_path); print(f"模型已保存为 {save_path}")
        except Exception as e: print(f"错误: 保存模型失败: {e}")

    def load_existing_model(self):
        load_path = self.model_path
        if not os.path.exists(load_path) and load_path.endswith(".h5"):
            keras_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_path): load_path = keras_path
        elif not os.path.exists(load_path) and not load_path.endswith(".keras"):
            keras_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_path): load_path = keras_path
            elif os.path.exists(self.model_path): load_path = self.model_path
            else: print(f"模型文件 {self.model_path} (或 .keras) 未找到。"); return None
        elif not os.path.exists(load_path): print(f"模型文件 {load_path} 未找到。"); return None
        print(f"加载已保存的模型：{load_path}")
        try:
            model = load_model(load_path, custom_objects={'PositionalEncoding': PositionalEncoding}) 
            if model.input_shape[1] != self.sequence_length or model.input_shape[2] != len(self.feature_names):
                print(f"警告: 加载的模型输入形状 ({model.input_shape}) 与当前配置 (seq_len={self.sequence_length}, num_features={len(self.feature_names)}) 不匹配。将忽略。")
                return None
            if model.output_shape[-1] != len(self.target_names):
                print(f"警告: 加载的模型输出维度与当前目标数不匹配。将忽略。")
                return None
            print("已加载模型与当前配置兼容。")
            return model
        except Exception as e: print(f"错误: 加载模型 {load_path} 失败: {e}。"); return None

    def tune_model(self):
        tuner_base_dir = tempfile.mkdtemp(prefix="aqi_tuner_")
        print(f"Keras Tuner 将使用临时目录: {tuner_base_dir}")
        try:
            def build_tuned_model(hp):
                d_model = hp.Choice('d_model', [64, 128, 192, 256]) 
                num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3, step=1) 
                num_heads = hp.Choice('num_heads', [4, 8]) 
                if d_model % num_heads != 0: 
                    num_heads = 4 if d_model % 4 == 0 else (2 if d_model % 2 == 0 else 1) 
                
                ff_dim_factor = hp.Choice('ff_dim_factor', [2, 4])
                ff_dim = d_model * ff_dim_factor
                dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.05) 
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG', default=5e-4) 
                model = self.build_model(d_model=d_model, num_transformer_blocks=num_transformer_blocks,
                                         num_heads=num_heads, ff_dim=ff_dim, dropout_rate=dropout_rate, 
                                         learning_rate=learning_rate) 
                return model

            tuner = BayesianOptimization(build_tuned_model, objective='val_loss', max_trials=20, executions_per_trial=1, 
                                         directory=tuner_base_dir, project_name='aqi_transformer_optimization_v5', overwrite=True)
            early_stopping_tuner = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
            tuner.search(self.X_train, self.y_train, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping_tuner])
            
            best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hp_list: print("警告: Tuner 未找到最佳超参数。"); return self.build_model()
            best_hp = best_hp_list[0]
            print(f"最佳超参数: d_model={best_hp.get('d_model')}, blocks={best_hp.get('num_transformer_blocks')}, heads={best_hp.get('num_heads')}, ff_factor={best_hp.get('ff_dim_factor')}, dropout={best_hp.get('dropout')}, lr={best_hp.get('learning_rate')}")
            best_models = tuner.get_best_models(num_models=1)
            if not best_models: print("警告: Tuner 未能构建最佳模型。"); return self.build_model()
            optimized_model = best_models[0]
            if not optimized_model.optimizer:
                optimized_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hp.get('learning_rate')), loss='mean_squared_error', metrics=['mae'])
            return optimized_model
        except Exception as e: print(f"Keras Tuner 调优出错: {e}"); return self.build_model()
        finally:
            if os.path.exists(tuner_base_dir):
                try: shutil.rmtree(tuner_base_dir)
                except Exception as e_clean: print(f"警告: 清理 Tuner 目录失败: {e_clean}")

    def predict_future(self, model, city_name, output_dir, num_steps=72):
        if self.last_sequence_for_future_pred_scaled is None:
            print("错误: 没有用于初始化未来预测的序列。")
            return None
        
        current_input_sequence_scaled = self.last_sequence_for_future_pred_scaled.copy() # (seq_len, num_input_features)
        
        # This history_df will store the *target variables* (unscaled) for calculating future features
        history_df = self.initial_history_df_for_future_features.copy()

        future_predictions_all_targets_inversed_list = [] # Store unscaled predictions
        
        print(f"\n开始自回归预测未来 {num_steps} 小时的所有指标...")
        
        start_dt = self.true_absolute_last_datetime
        if start_dt is None: start_dt = pd.Timestamp.now().replace(minute=0,second=0,microsecond=0)
        
        future_timestamps = pd.date_range(start=start_dt + pd.Timedelta(hours=1), periods=num_steps, freq='h')
        print(f"信息: 未来预测将从 {future_timestamps[0]} 开始，直到 {future_timestamps[-1]}。")

        for i in range(num_steps):
            current_input_reshaped = current_input_sequence_scaled.reshape(1, self.sequence_length, len(self.feature_names))
            pred_step_all_targets_scaled = model.predict(current_input_reshaped, verbose=0)[0]
            
            # Inverse transform the prediction to get unscaled target values
            pred_step_all_targets_inversed = self.scaler_y.inverse_transform(pred_step_all_targets_scaled.reshape(1,-1))[0]
            future_predictions_all_targets_inversed_list.append(pred_step_all_targets_inversed)
            
            # Create a new row for the history_df with the predicted target values
            new_history_row_data = {name: pred_step_all_targets_inversed[j] for j, name in enumerate(self.target_names)}
            new_history_row_data['datetime'] = future_timestamps[i]
            new_history_row_data[self.hour_col_name] = future_timestamps[i].hour
            new_history_row_df = pd.DataFrame([new_history_row_data])
            
            history_df = pd.concat([history_df, new_history_row_df[['datetime', self.hour_col_name] + self.target_names]], ignore_index=True)
            # Keep only the necessary window for the next feature calculation
            history_df = history_df.iloc[-(self.sequence_length + max(self.lag_steps, max(self.rolling_windows))):].copy()

            # Prepare the features for the *next* input sequence using the updated history_df
            df_for_next_features_calc = history_df.iloc[-self.sequence_length:].copy() # Get the latest sequence_length part
            df_for_next_features_calc = self._add_time_features(df_for_next_features_calc)
            df_for_next_features_calc = self._add_lagged_features(df_for_next_features_calc, self.target_names, self.lag_steps)
            cols_for_rolling = ['PM2.5', 'PM10', 'O3', 'AQI']
            df_for_next_features_calc = self._add_rolling_features(df_for_next_features_calc, [c for c in cols_for_rolling if c in df_for_next_features_calc.columns], self.rolling_windows)
            
            # Handle potential NaNs in the newly created features for the last row
            df_for_next_features_calc.ffill(inplace=True) # Fill NaNs that might appear
            df_for_next_features_calc.bfill(inplace=True) # Fill any remaining at the start

            next_input_features_unscaled_vector = df_for_next_features_calc[self.feature_names].iloc[-1].values
            next_input_features_scaled_vector = self.scaler_X.transform(next_input_features_unscaled_vector.reshape(1, -1)).flatten()
            
            current_input_sequence_scaled = np.roll(current_input_sequence_scaled, -1, axis=0)
            current_input_sequence_scaled[-1, :] = next_input_features_scaled_vector
        
        future_predictions_all_targets_inversed_np = np.array(future_predictions_all_targets_inversed_list)
        df_future = pd.DataFrame(future_predictions_all_targets_inversed_np, columns=self.target_names)
        df_future.insert(0, 'Timestamp', future_timestamps)
        
        df_excel = df_future.copy()
        iaqi_cols_added = []
        for model_pred_col, iaqi_calc_key in MODEL_TO_IAQI_POLLUTANT_MAP.items():
            if model_pred_col in df_excel.columns:
                iaqi_col_name = f"IAQI_{iaqi_calc_key.split('_')[0]}"
                df_excel[iaqi_col_name] = df_excel[model_pred_col].apply(lambda x: calculate_iaqi(x, iaqi_calc_key))
                iaqi_cols_added.append(iaqi_col_name)
        if iaqi_cols_added:
            df_excel['AQI_calculated'] = df_excel[iaqi_cols_added].max(axis=1)
            primary_pollutants = []
            for _, row in df_excel.iterrows():
                max_val = row['AQI_calculated']
                pps = [col.replace("IAQI_","") for col in iaqi_cols_added if pd.notna(row[col]) and round(row[col]) == round(max_val)]
                primary_pollutants.append(", ".join(pps) if pps else "N/A")
            df_excel['PrimaryPollutant'] = primary_pollutants
        else: df_excel['AQI_calculated'], df_excel['PrimaryPollutant'] = np.nan, "N/A"

        cols_to_round_excel = self.target_names + iaqi_cols_added + ['AQI_calculated']
        for col in cols_to_round_excel:
            if col in df_excel.columns: df_excel[col] = df_excel[col].round().astype('Int64', errors='ignore') 

        df_excel['date'] = df_excel['Timestamp'].dt.strftime('%Y%m%d')
        df_excel['hour'] = df_excel['Timestamp'].dt.hour
        excel_cols = ['date', 'hour'] + [c for c in self.target_names if c in df_excel.columns] + \
                       [c for c in iaqi_cols_added if c in df_excel.columns] + \
                       [c for c in ['AQI_calculated', 'PrimaryPollutant'] if c in df_excel.columns]
        df_excel_final = df_excel[excel_cols]

        print("\n未来3天每小时所有指标预测 (自回归，含计算后AQI):"); print(df_excel_final.to_string())
        forecast_file = os.path.join(output_dir, f"{city_name}_AQI_Data_forecast.xlsx")
        try: df_excel_final.to_excel(forecast_file, index=False); print(f"预测结果已保存到: {forecast_file}")
        except Exception as e: print(f"错误: 保存预测结果失败: {e}")
        
        num_plot_targets = len(self.target_names) + (1 if 'AQI_calculated' in df_excel.columns else 0)
        fig, axes = plt.subplots(num_plot_targets, 1, figsize=(12, 3 * num_plot_targets), sharex=True)
        if num_plot_targets == 1: axes = [axes]
        fig.suptitle('未来3天每小时各项指标预测 (自回归)', fontsize=16, y=1.02)
        plot_list = self.target_names + (['AQI_calculated'] if 'AQI_calculated' in df_excel.columns else [])

        for i, target_col in enumerate(plot_list):
            ax = axes[i]
            plot_series = df_future[target_col] if target_col in df_future.columns else df_excel[target_col] 
            ax.plot(future_timestamps, plot_series, label=f'预测 {target_col}', color='purple', marker='.', ls='-')
            ax.set_ylabel(target_col, fontsize=10); ax.legend(loc='upper right', fontsize=8); ax.grid(True, ls='--', alpha=0.7)
        axes[-1].set_xlabel('时间', fontsize=12); plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')); plt.xticks(rotation=30, ha='right'); plt.tight_layout(rect=[0,0,1,0.98]); plt.show()
        return df_future 

    def evaluate(self, y_true_multi, y_pred_multi): 
        if y_true_multi is None or y_pred_multi is None:
            return {target: {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')} for target in self.target_names}
        all_metrics_eval = {}
        num_targets = y_true_multi.shape[1]
        target_names_to_use = self.target_names[:num_targets] if num_targets != len(self.target_names) else self.target_names
        if num_targets != len(self.target_names): print(f"警告: 评估时目标数量 ({num_targets}) 与预期 ({len(self.target_names)}) 不匹配。")

        for i in range(num_targets):
            target_name = target_names_to_use[i]
            y_true, y_pred = y_true_multi[:, i].flatten(), y_pred_multi[:, i].flatten()
            valid_mask = ~np.isnan(y_pred) & ~np.isinf(y_pred) & ~np.isnan(y_true) & ~np.isinf(y_true)
            if not np.any(valid_mask): 
                all_metrics_eval[target_name] = {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')}
                continue
            y_true_f, y_pred_f = y_true[valid_mask], y_pred[valid_mask]
            if len(y_true_f) == 0: 
                 all_metrics_eval[target_name] = {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')}
                 continue
            epsilon = 1e-8 
            mape_values = np.abs((y_true_f - y_pred_f) / (y_true_f + epsilon))
            mape = np.nanmean(np.where(np.isinf(mape_values), np.nan, mape_values)) * 100 
            mape = min(mape, 1000.0) if pd.notna(mape) else float('nan')
            all_metrics_eval[target_name] = {
                "MAPE": mape, "RMSE": np.sqrt(mean_squared_error(y_true_f, y_pred_f)),
                "R2": r2_score(y_true_f, y_pred_f), "MAE": mean_absolute_error(y_true_f, y_pred_f)
            }
        return all_metrics_eval
    
    def compare_forecast_with_truth_and_trigger(self, df_forecast, future_ground_truth_path, deviation_threshold=0.20, points_threshold_percentage=0.30):
        if df_forecast is None or df_forecast.empty: print("警告 (compare): 预测DataFrame为空。"); return False 
        if 'Timestamp' not in df_forecast.columns:
            df_forecast = df_forecast.reset_index() if 'Timestamp' in df_forecast.index.names else None
            if df_forecast is None or 'Timestamp' not in df_forecast.columns: print(f"错误 (compare): 'Timestamp' 列缺失。"); return False
        if not future_ground_truth_path or not os.path.exists(future_ground_truth_path): print(f"警告: 未找到未来真实值文件: {future_ground_truth_path}。"); return False 
        try:
            df_truth_raw = pd.read_excel(future_ground_truth_path)
            if self.date_col_name not in df_truth_raw.columns or self.hour_col_name not in df_truth_raw.columns: print(f"错误: 未来真实值文件缺少 'date'/'hour'。"); return False
            df_truth_raw['datetime_str'] = df_truth_raw[self.date_col_name].astype(str) + df_truth_raw[self.hour_col_name].astype(str).str.zfill(2)
            df_truth_raw['Timestamp'] = pd.to_datetime(df_truth_raw['datetime_str'], format='%Y%m%d%H')
            cols_for_truth = ['Timestamp'] + self.target_names
            missing_truth_cols = [col for col in cols_for_truth if col not in df_truth_raw.columns]
            if missing_truth_cols: print(f"错误: 未来真实值文件缺少列: {missing_truth_cols}"); return False
            df_truth = df_truth_raw[cols_for_truth].copy()
        except Exception as e: print(f"错误: 处理未来真实值文件失败: {e}"); return False
        merged_df = pd.merge(df_forecast, df_truth, on='Timestamp', suffixes=('_pred', '_true'), how='inner')
        if merged_df.empty: print("警告: 预测与真实值无重叠数据。"); return False
        triggered = False
        for target_col in self.target_names:
            pred_col, true_col = target_col + '_pred', target_col + '_true' 
            if pred_col not in merged_df.columns: pred_col = target_col 
            
            if pred_col not in merged_df.columns or true_col not in merged_df.columns:
                print(f"警告: 指标 '{target_col}' 在合并DF中未找到预测/真实列。跳过。")
                continue
            y_pred, y_true = merged_df[pred_col].astype(float).values, merged_df[true_col].astype(float).values
            valid = ~np.isnan(y_pred)&~np.isinf(y_pred)&~np.isnan(y_true)&~np.isinf(y_true)
            if not np.any(valid): print(f"警告: 指标 '{target_col}' (未来对比) 无有效数据。"); continue
            y_true_f, y_pred_f = y_true[valid], y_pred[valid]
            if len(y_true_f) == 0: print(f"警告: 指标 '{target_col}' (未来对比) 无有效对。"); continue
            deviations = np.abs(y_pred_f - y_true_f) / (y_true_f + 1e-8)
            perc_exceed = np.sum(deviations > deviation_threshold) / len(y_true_f)
            print(f"信息 (未来对比): '{target_col}' 偏差>阈值百分比: {perc_exceed*100:.2f}% (触发阈值: {points_threshold_percentage*100:.0f}%)")
            if perc_exceed >= points_threshold_percentage: print(f"触发: '{target_col}' 预测性能不佳。"); triggered = True
        return triggered

    def identify_prediction_anomalies(self, model, X_data_scaled, y_data_scaled, dates, threshold_multiplier=3):
        print("\n--- 开始检测预测异常点 (基于评估测试集) ---")
        if X_data_scaled is None or y_data_scaled is None or dates is None or len(X_data_scaled) == 0:
            print("警告: 无法执行异常检测，缺少数据。")
            return

        predictions_scaled = model.predict(X_data_scaled)
        predictions_inversed = self.scaler_y.inverse_transform(predictions_scaled)
        y_true_inversed = self.scaler_y.inverse_transform(y_data_scaled)

        anomaly_reports = []

        for i, target_name in enumerate(self.target_names):
            true_vals = y_true_inversed[:, i]
            pred_vals = predictions_inversed[:, i]
            
            errors = np.abs(true_vals - pred_vals)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            threshold = mean_error + threshold_multiplier * std_error

            anomalous_indices = np.where(errors > threshold)[0]
            
            if len(anomalous_indices) > 0:
                print(f"\n指标 '{target_name}' 的预测异常点 (误差 > {threshold:.2f}):")
                for idx in anomalous_indices:
                    report_str = (f"  时间: {dates.iloc[idx]}, "
                                  f"真实值: {true_vals[idx]:.2f}, "
                                  f"预测值: {pred_vals[idx]:.2f}, "
                                  f"误差: {errors[idx]:.2f}")
                    print(report_str)
                    anomaly_reports.append(report_str)
            else:
                print(f"\n指标 '{target_name}': 未检测到显著预测异常点。")
        return anomaly_reports


    def plot_results(self, y_true_multi, y_pred_base_multi, y_pred_opt_multi):
        if y_true_multi is None: return
        targets_to_plot = ['AQI', 'PM2.5', 'PM10'] 
        plot_indices = [self.target_names.index(name) for name in targets_to_plot if name in self.target_names]
        if not plot_indices: return
        num_plots = len(plot_indices)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)
        if num_plots == 1: axes = [axes]
        fig.suptitle('部分指标预测结果对比', fontsize=16, y=1.02 if num_plots > 1 else 1.05)
        for k, target_idx in enumerate(plot_indices):
            ax, name = axes[k], self.target_names[target_idx]
            ax.plot(self.test_dates, y_true_multi[:, target_idx].flatten(), label=f'实际 {name}', color='blue', lw=1)
            if y_pred_base_multi is not None:
                base_preds = y_pred_base_multi[:, target_idx].flatten()
                valid_mask = ~np.isnan(base_preds)
                if np.any(valid_mask): ax.plot(self.test_dates[valid_mask], base_preds[valid_mask], label=f'基线预测 {name}', color='orange', alpha=0.7, ls='--')
            if y_pred_opt_multi is not None:
                opt_preds = y_pred_opt_multi[:, target_idx].flatten()
                valid_mask = ~np.isnan(opt_preds)
                if np.any(valid_mask): ax.plot(self.test_dates[valid_mask], opt_preds[valid_mask], label=f'优化预测 {name}', color='green', alpha=0.7, ls=':')
            ax.set_ylabel(name, fontsize=10); ax.legend(loc='upper right', fontsize=8); ax.grid(True, ls='--', alpha=0.7)
        if pd.api.types.is_datetime64_any_dtype(self.test_dates):
             axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')) 
             plt.xticks(rotation=30, ha='right') 
        axes[-1].set_xlabel('时间', fontsize=12); plt.tight_layout(rect=[0,0,1,0.97 if num_plots>1 else 0.95]); plt.show()

    def plot_metrics_comparison(self, eval_baseline, eval_optimized):
        metrics_plot, targets_plot = ['MAPE', 'RMSE', 'R2', 'MAE'], ['AQI', 'PM2.5']
        fig, axes = plt.subplots(len(metrics_plot), 1, figsize=(10, 3 * len(metrics_plot)), sharex=False)
        if len(metrics_plot) == 1: axes = [axes]
        fig.suptitle('关键指标的评估对比', fontsize=16, y=1.02)
        for i, metric_name in enumerate(metrics_plot):
            ax, base_vals, opt_vals, labels = axes[i], [], [], []
            for target in targets_plot:
                if target in eval_baseline and target in eval_optimized:
                    base_vals.append(eval_baseline[target].get(metric_name, 0 if metric_name != 'MAPE' else 1000))
                    opt_vals.append(eval_optimized[target].get(metric_name, 0 if metric_name != 'MAPE' else 1000))
                    labels.append(target)
            if not labels: continue
            x, width = np.arange(len(labels)), 0.35
            r1, r2 = ax.bar(x - width/2, base_vals, width, label='Baseline', color='skyblue'), ax.bar(x + width/2, opt_vals, width, label='Optimized', color='lightcoral')
            ax.set_ylabel(metric_name, fontsize=10); ax.set_title(f'{metric_name} 对比', fontsize=12)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0, fontsize=9); ax.legend(fontsize=8); ax.grid(axis='y', ls='--', alpha=0.7)
            def autolabel(rects, evals):
                for k_rect, rect in enumerate(rects):
                    orig_val = evals.get(labels[k_rect], {}).get(metric_name, float('nan'))
                    txt = f"{orig_val:.2f}" if pd.notna(orig_val) else "N/A"
                    ax.annotate(txt, xy=(rect.get_x() + rect.get_width()/2, rect.get_height()), xytext=(0,3), textcoords="offset points", ha='center', va='bottom', fontsize=7)
            autolabel(r1, eval_baseline); autolabel(r2, eval_optimized)
        plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

def select_data_file_path(title="请选择数据集文件"):
    root = tk.Tk(); root.withdraw(); path = filedialog.askopenfilename(title=title, filetypes=[("Excel", "*.xlsx"), ("All", "*.*")]); root.destroy(); return path
def select_model_file_path(title="选择模型文件路径", default_name="model.keras"):
    root = tk.Tk(); root.withdraw(); path = filedialog.asksaveasfilename(title=title, initialfile=default_name, defaultextension=".keras", filetypes=[("Keras", "*.keras"), ("HDF5", "*.h5"), ("All", "*.*")]); root.destroy(); return path
def select_verification_data_path(title="请选择验证/测试数据集 (例如 南京_AQI_Data_test.xlsx)"):
    root = tk.Tk(); root.withdraw(); path = filedialog.askopenfilename(title=title, filetypes=[("Excel", "*.xlsx"), ("All", "*.*")]); root.destroy(); return path


def main(): 
    root = tk.Tk(); root.withdraw() 
    predict_future_choice = messagebox.askyesno("未来预测", "您想使用模型预测未来3天每小时的所有指标吗？")
    train_model_choice = messagebox.askyesno("模型训练", "您想训练或重新训练预测模型吗？")
    root.destroy() 

    data_path = select_data_file_path("请选择训练/验证数据集 Excel 文件")
    if not data_path: print("未选择训练数据文件，程序退出。"); return

    verification_data_path = select_verification_data_path(
        "请选择验证/测试数据集 (例如 南京_AQI_Data_test.xlsx)\n此文件将用于确定城市名、预测输出目录，并作为未来预测的真实值参考。"
    )
    if not verification_data_path:
        messagebox.showwarning("警告", "未选择验证/测试数据集。未来预测文件名和路径将使用默认值，且无法进行基于未来预测的自动重训练。")
        city_name_for_forecast, forecast_output_dir = "DefaultCity", "."
    else:
        forecast_output_dir = os.path.dirname(verification_data_path)
        base_name = os.path.basename(verification_data_path)
        match = re.match(r"([^_]+)_AQI_Data", base_name, re.UNICODE)
        city_name_for_forecast = match.group(1) if match else os.path.splitext(base_name)[0].split('_')[0]
        print(f"未来预测将使用城市名: '{city_name_for_forecast}' 并保存在目录: '{forecast_output_dir}'")


    model_path = select_model_file_path("指定模型文件的加载/保存路径", f"{city_name_for_forecast}_transformer_model.keras")
    if not model_path: print("未指定模型文件路径，程序退出。"); return
    
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        try: os.makedirs(model_dir); print(f"已创建模型目录: {model_dir}")
        except OSError as e: print(f"错误: 创建模型目录 {model_dir} 失败: {e}。"); model_path = os.path.basename(model_path)

    try: predictor = AQIPredictor(data_path, model_path)
    except Exception as e: print(f"初始化AQIPredictor失败: {e}"); return

    model_to_use, y_test_eval_final, initial_preds_on_eval_set, final_preds_on_eval_set = None, None, None, None
    max_retrain_attempts, current_attempt = 2, 0 
    run_user_optimization = False # Initialize run_user_optimization
    
    static_target_names = predictor.target_names
    eval_template = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}
    eval_initial_model = {target: eval_template.copy() for target in static_target_names} 
    eval_final_model = {target: eval_template.copy() for target in static_target_names} 

    X_eval_set, y_eval_set_scaled, dates_eval_set = predictor.X_test, predictor.y_test, predictor.test_dates
    if X_eval_set is None or y_eval_set_scaled is None or X_eval_set.size == 0:
        messagebox.showerror("错误", "没有可用的内部测试数据进行模型评估。程序将退出。")
        return
    y_test_eval_final = predictor.scaler_y.inverse_transform(y_eval_set_scaled) 
    predictor.test_dates = dates_eval_set 


    if train_model_choice:
        model_to_use = predictor.load_existing_model()
        if model_to_use:
            print("重新训练已加载的基线模型 (微调)...")
            opt = keras.optimizers.Adam(learning_rate=1e-5); model_to_use.compile(optimizer=opt, loss='msle', metrics=['mae'])
            initial_preds_on_eval_set, _ = predictor.train_and_evaluate(model_to_use, epochs=30, X_test_override=X_eval_set, y_test_override=y_eval_set_scaled)
        else:
            print("构建并训练新的基线模型...")
            model_to_use = predictor.build_model(d_model=128, num_transformer_blocks=2, num_heads=4, ff_dim=256, learning_rate=0.001)
            initial_preds_on_eval_set, _ = predictor.train_and_evaluate(model_to_use, epochs=150, X_test_override=X_eval_set, y_test_override=y_eval_set_scaled)
        predictor.save_model(model_to_use)
    else:
        model_to_use = predictor.load_existing_model()
        if not model_to_use: messagebox.showerror("错误", "未找到模型且未选择训练。"); return
        if not model_to_use.optimizer: model_to_use.compile(optimizer=keras.optimizers.Adam(1e-4), loss='msle', metrics=['mae'])
        preds_scaled = model_to_use.predict(X_eval_set)
        initial_preds_on_eval_set = predictor.scaler_y.inverse_transform(preds_scaled)

    if y_test_eval_final is not None and initial_preds_on_eval_set is not None:
        eval_initial_model = predictor.evaluate(y_test_eval_final, initial_preds_on_eval_set)
    
    final_model_for_future_prediction = model_to_use 
    eval_final_model = eval_initial_model.copy() 
    final_preds_on_eval_set_for_plot = initial_preds_on_eval_set 
    optimization_attempted_and_succeeded = False 

    if predict_future_choice: 
        for attempt in range(max_retrain_attempts + 1):
            print(f"\n--- 未来预测与评估迭代: {attempt + 1} ---")
            df_future = predictor.predict_future(final_model_for_future_prediction, city_name=city_name_for_forecast, output_dir=forecast_output_dir, num_steps=72)
            if df_future is None: messagebox.showerror("错误", "未来预测失败，无法继续迭代。"); break
            if not verification_data_path: print("未提供未来真实值验证文件，跳过基于未来预测的重训练触发。"); break
            needs_retrain_based_on_future = predictor.compare_forecast_with_truth_and_trigger(df_future, verification_data_path)
            if needs_retrain_based_on_future and attempt < max_retrain_attempts:
                print(f"未来预测准确性不足 (尝试 {attempt + 1})。开始重新优化和训练模型...")
                tuned_model_candidate = predictor.tune_model()
                if tuned_model_candidate:
                    final_model_for_future_prediction = tuned_model_candidate 
                    print("训练新调优的模型...")
                    current_optimized_preds, _ = predictor.train_and_evaluate(final_model_for_future_prediction, epochs=150, X_test_override=X_eval_set, y_test_override=y_eval_set_scaled)
                    predictor.save_model(final_model_for_future_prediction) 
                    if y_test_eval_final is not None and current_optimized_preds is not None:
                         eval_final_model = predictor.evaluate(y_test_eval_final, current_optimized_preds)
                         final_preds_on_eval_set_for_plot = current_optimized_preds 
                    optimization_attempted_and_succeeded = True
                else: print("超参数调优未能产生新模型，停止迭代。"); break
            else: 
                if needs_retrain_based_on_future: print("已达到最大自动重训练尝试次数，使用当前模型。")
                else: print("未来预测准确性可接受。")
                break 
    elif train_model_choice: 
        root = tk.Tk(); root.withdraw()
        run_user_optimization = messagebox.askyesno("模型优化", "初始模型已训练/加载。\n您想运行贝叶斯优化以尝试改进模型吗？\n（这可能需要一些时间）")
        root.destroy()
        if run_user_optimization:
            print("\n--- 用户选择优化模型 ---")
            tuned_model_candidate = predictor.tune_model()
            if tuned_model_candidate:
                final_model_for_future_prediction = tuned_model_candidate
                print("训练和评估用户选择优化后的模型...")
                final_preds_on_eval_set_for_plot, _ = predictor.train_and_evaluate(final_model_for_future_prediction, epochs=150, X_test_override=X_eval_set, y_test_override=y_eval_set_scaled)
                predictor.save_model(final_model_for_future_prediction)
                if y_test_eval_final is not None and final_preds_on_eval_set_for_plot is not None:
                    eval_final_model = predictor.evaluate(y_test_eval_final, final_preds_on_eval_set_for_plot)
                optimization_attempted_and_succeeded = True
            else: print("用户选择的优化失败。")

    # Anomaly detection on the evaluation set using the final model
    if final_model_for_future_prediction and X_eval_set is not None and y_eval_set_scaled is not None and dates_eval_set is not None:
        print("\n--- 在评估测试集上进行预测异常点检测 ---")
        predictor.identify_prediction_anomalies(final_model_for_future_prediction, X_eval_set, y_eval_set_scaled, dates_eval_set)


    print("\n--- 最终评估结果 ---")
    print("\n初始模型在评估测试集上的性能:")
    for target_name, metrics in eval_initial_model.items():
        print(f"  指标 {target_name}:"); [print(f"    {m}: {v:.4f}") for m, v in metrics.items()]
    
    if optimization_attempted_and_succeeded:
        print("\n最终优化模型在评估测试集上的性能:")
        for target_name, metrics in eval_final_model.items():
            print(f"  指标 {target_name}:"); [print(f"    {m}: {v:.4f}") for m, v in metrics.items()]
    elif train_model_choice : 
         print("\n未执行有效优化或用户未选择优化，最终模型性能同初始训练/加载模型。")
    elif not train_model_choice :
         print("\n仅加载模型，未进行训练或优化。")


    print("\n--- 生成图表 (基于评估测试集) ---")
    if y_test_eval_final is not None: 
        plot_optimized_preds = final_preds_on_eval_set_for_plot if optimization_attempted_and_succeeded else None
        predictor.plot_results(y_test_eval_final, initial_preds_on_eval_set, plot_optimized_preds)
        predictor.plot_metrics_comparison(eval_initial_model, eval_final_model if optimization_attempted_and_succeeded else eval_initial_model)
    else: print("警告: 由于缺少测试集真实值，无法生成图表。")

    if predict_future_choice:
        if final_model_for_future_prediction: 
            print("\n--- 使用最终确定的模型进行未来所有指标预测 ---")
            predictor.predict_future(final_model_for_future_prediction, city_name=city_name_for_forecast, output_dir=forecast_output_dir, num_steps=72)
        else: 
            messagebox.showwarning("未来预测", "没有可用的模型进行未来预测。")
            print("没有可用的模型进行未来预测。")

if __name__ == "__main__":
    main()
