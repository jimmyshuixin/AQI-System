import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from keras import Input, Model
from keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, Flatten, Embedding, Add, Concatenate, Conv1D 
from keras_tuner import BayesianOptimization
import matplotlib.pyplot as plt
import os
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog, messagebox
import tempfile
import shutil
from tensorflow import keras
import tensorflow as tf
import re
import math
import glob 

# --- Matplotlib Configuration for Chinese Characters ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        print("警告: 未能设置中文字体。图表中的中文可能无法正确显示。")
# --- End Matplotlib Configuration ---

# --- IAQI Calculation Constants ---
IAQI_LEVELS = [0, 50, 100, 150, 200, 300, 400, 500]
POLLUTANT_BREAKPOINTS = {
    'SO2_24h': [0, 50, 150, 475, 800, 1600, 2100, 2620], 
    'NO2_24h': [0, 40, 80, 180, 280, 565, 750, 940],    
    'PM10_24h': [0, 50, 150, 250, 350, 420, 500, 600],  
    'CO_24h': [0, 2, 4, 14, 24, 36, 48, 60],          
    'O3_8h': [0, 100, 160, 215, 265, 800],            
    'PM2.5_24h': [0, 35, 75, 115, 150, 250, 350, 500]   
}
O3_IAQI_LEVELS_FOR_CALC = [0, 50, 100, 150, 200, 300] 

MODEL_TO_IAQI_POLLUTANT_MAP = {
    'SO2_24h': 'SO2_24h', 'NO2_24h': 'NO2_24h', 'PM10_24h': 'PM10_24h',
    'CO_24h': 'CO_24h', 'O3_8h': 'O3_8h', 'PM2.5_24h': 'PM2.5_24h'
}

def calculate_iaqi(c_p, pollutant_key):
    if pollutant_key not in POLLUTANT_BREAKPOINTS: return np.nan
    bp_list = POLLUTANT_BREAKPOINTS[pollutant_key]
    iaqi_levels_for_pollutant_calc = O3_IAQI_LEVELS_FOR_CALC if pollutant_key == 'O3_8h' else IAQI_LEVELS

    if pd.isna(c_p) or c_p < 0: return np.nan
    for i in range(len(bp_list) - 1):
        bp_lo, bp_hi = bp_list[i], bp_list[i+1]
        if i + 1 < len(iaqi_levels_for_pollutant_calc):
            iaqi_lo, iaqi_hi = iaqi_levels_for_pollutant_calc[i], iaqi_levels_for_pollutant_calc[i+1]
            if bp_lo <= c_p <= bp_hi:
                return round(((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo)) * (c_p - bp_lo) + iaqi_lo) if bp_hi != bp_lo else iaqi_lo
        else: break 
            
    if c_p > bp_list[-1]:
        if pollutant_key == 'O3_8h': return IAQI_LEVELS[-1] 
        return iaqi_levels_for_pollutant_calc[-1]
    
    if pollutant_key == 'O3_8h' and len(bp_list) > 1 and bp_list[-2] <= c_p <= bp_list[-1]:
         iaqi_lo, iaqi_hi = O3_IAQI_LEVELS_FOR_CALC[-2], O3_IAQI_LEVELS_FOR_CALC[-1]
         bp_lo_o3, bp_hi_o3 = POLLUTANT_BREAKPOINTS['O3_8h'][-2] , POLLUTANT_BREAKPOINTS['O3_8h'][-1]
         if bp_lo_o3 <= c_p <= bp_hi_o3 : 
            return round(((iaqi_hi - iaqi_lo) / (bp_hi_o3 - bp_lo_o3)) * (c_p - bp_lo_o3) + iaqi_lo) if bp_hi_o3 != bp_lo_o3 else iaqi_lo
    return np.nan


class PositionalEncoding(keras.layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.pos_emb = Embedding(input_dim=sequence_length, output_dim=d_model)

    def call(self, x): 
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1) 
        embedded_positions = self.pos_emb(positions) 
        return x + embedded_positions 

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "d_model": self.d_model})
        return config

class WarmupCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, total_steps, warmup_steps, initial_lr, target_lr_min=1e-6, verbose=0):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.total_steps = total_steps; self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr; self.target_lr_min = target_lr_min
        self.verbose = verbose; self.current_step = 0
    def on_batch_begin(self, batch, logs=None):
        self.current_step += 1
        if self.current_step <= self.warmup_steps: lr = self.initial_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = (self.initial_lr - self.target_lr_min) * cosine_decay + self.target_lr_min
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        if self.verbose > 0 and (self.current_step % 100 == 0 or self.current_step == 1): print(f"\nStep {self.current_step}: LR set to {lr:.7f}.")

class AQIPredictor:
    def __init__(self, root_data_folder_path, model_path="trained_model.h5", sequence_length=24, lag_steps=3, rolling_windows=[3, 6]):
        self.root_data_folder_path = root_data_folder_path 
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.lag_steps = lag_steps
        self.rolling_windows = rolling_windows
        
        self.base_feature_names = ['PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 'CO', 'CO_24h']
        self.target_names = self.base_feature_names + ['AQI']
        if 'AQI' not in self.base_feature_names: 
            if 'AQI' not in self.target_names: self.target_names.append('AQI')
        
        self.feature_names = [] 
        self.site_id_col = "site_id_numeric" 
        self.site_name_col = "site_name" 

        self.scaler_X = MinMaxScaler() 
        self.scaler_y = MinMaxScaler() 
        
        self.date_col_name = 'date' 
        self.hour_col_name = 'hour' 
        self.true_absolute_last_datetime_map = {} 
        self.last_sequence_for_future_pred_scaled_map = {} 
        self.initial_history_df_for_future_features_map = {} 
        self.site_to_id = {}
        self.id_to_site = {}
        self.num_sites = 0
        
        self._load_and_prepare_data() 
    
    def _add_time_features(self, df_input):
        df = df_input.copy()
        if 'datetime' not in df.columns:
            if self.date_col_name in df.columns and self.hour_col_name in df.columns:
                 df['datetime_str_temp'] = df[self.date_col_name].astype(str) + df[self.hour_col_name].astype(str).str.zfill(2)
                 df['datetime'] = pd.to_datetime(df['datetime_str_temp'], format='%Y%m%d%H', errors='coerce')
                 df.dropna(subset=['datetime'], inplace=True) 
            else:
                raise ValueError("DataFrame for time feature creation must contain 'datetime' column or 'date' and 'hour' columns.")

        df['hour_sin'] = np.sin(2 * np.pi * df[self.hour_col_name].astype(float)/23.0) 
        df['hour_cos'] = np.cos(2 * np.pi * df[self.hour_col_name].astype(float)/23.0)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofweek/6.0) 
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofweek/6.0)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['datetime'].dt.dayofyear/365.0) 
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['datetime'].dt.dayofyear/365.0)
        return df

    def _add_lagged_features(self, df_input, cols_to_lag, lag_steps):
        df_lagged = df_input.copy()
        for col in cols_to_lag:
            if col in df_lagged.columns:
                for lag in range(1, lag_steps + 1): df_lagged[f'{col}_lag{lag}'] = df_lagged[col].shift(lag)
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


    def _load_and_prepare_data(self):
        all_site_dfs_processed = [] 
        site_id_counter = 0
        
        file_pattern = os.path.join(self.root_data_folder_path, '**', 'china_cities_*.csv')
        excel_files = glob.glob(os.path.join(self.root_data_folder_path, '**', 'china_cities_*.xlsx'), recursive=True)
        csv_files = glob.glob(file_pattern, recursive=True)
        
        all_files = csv_files + excel_files
        if not all_files:
            raise FileNotFoundError(f"No CSV or XLSX files found in {self.root_data_folder_path} matching pattern 'china_cities_*'.")
        print(f"Found {len(all_files)} data files to process.")

        processed_feature_names_flag = False 

        for file_path in all_files:
            print(f"Processing file: {file_path}")
            try:
                if file_path.endswith(".csv"):
                    df_wide_raw = pd.read_csv(file_path, low_memory=False)
                elif file_path.endswith(".xlsx"):
                    df_wide_raw = pd.read_excel(file_path) 
                else:
                    print(f"Skipping unsupported file type: {file_path}")
                    continue
            except Exception as e: print(f"错误: 读取文件 {file_path} 失败: {e}"); continue

            if self.date_col_name not in df_wide_raw.columns or self.hour_col_name not in df_wide_raw.columns or 'type' not in df_wide_raw.columns:
                print(f"警告: 文件 {file_path} 缺少 'date', 'hour', 或 'type' 列，已跳过。"); continue
            
            city_columns = [col for col in df_wide_raw.columns if col not in [self.date_col_name, self.hour_col_name, 'type']]
            
            for city_name in city_columns:
                if city_name not in self.site_to_id:
                    self.site_to_id[city_name] = site_id_counter
                    self.id_to_site[site_id_counter] = city_name
                    site_id_counter += 1
                site_id_numeric = self.site_to_id[city_name]

                df_city_specific = df_wide_raw[[self.date_col_name, self.hour_col_name, 'type', city_name]].copy()
                df_city_specific.rename(columns={city_name: 'value'}, inplace=True)
                
                try:
                    df_city_pivoted = df_city_specific.pivot_table(index=[self.date_col_name, self.hour_col_name], columns='type', values='value').reset_index()
                except Exception as e: print(f"警告: 为城市 '{city_name}' 从文件 '{file_path}' pivot数据失败: {e}。已跳过。"); continue

                for target_col in self.target_names:
                    if target_col not in df_city_pivoted.columns: df_city_pivoted[target_col] = np.nan
                
                df_city_pivoted['datetime_str_temp'] = df_city_pivoted[self.date_col_name].astype(str) + df_city_pivoted[self.hour_col_name].astype(str).str.zfill(2)
                df_city_pivoted['datetime'] = pd.to_datetime(df_city_pivoted['datetime_str_temp'], format='%Y%m%d%H', errors='coerce')
                df_city_pivoted.dropna(subset=['datetime'], inplace=True)
                df_city_pivoted.sort_values(by='datetime', inplace=True)
                
                if df_city_pivoted.empty: continue
                
                self.true_absolute_last_datetime_map[site_id_numeric] = df_city_pivoted['datetime'].iloc[-1]

                aqi_data_site = df_city_pivoted.copy()
                for col in self.target_names:
                    if col in aqi_data_site.columns: aqi_data_site[col] = np.log1p(aqi_data_site[col])
                aqi_data_site = self._add_time_features(aqi_data_site)
                
                current_site_feature_names_temp = self.base_feature_names + ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'day_of_year_sin', 'day_of_year_cos']
                for target_col_lag in self.target_names:
                    for lag in range(1, self.lag_steps + 1): current_site_feature_names_temp.append(f'{target_col_lag}_lag{lag}')
                cols_for_rolling = ['PM2.5', 'PM10', 'O3', 'AQI']
                for col_roll in cols_for_rolling:
                    for window in self.rolling_windows:
                        current_site_feature_names_temp.append(f'{col_roll}_roll_mean_{window}h'); current_site_feature_names_temp.append(f'{col_roll}_roll_std_{window}h')
                
                if not processed_feature_names_flag: 
                    self.feature_names = list(dict.fromkeys(current_site_feature_names_temp))
                    print(f"信息: 最终使用的输入特征 ({len(self.feature_names)}): {self.feature_names}")
                    processed_feature_names_flag = True

                cols_to_impute_log = self.target_names
                nan_check_log = aqi_data_site[cols_to_impute_log].isnull().sum()
                if nan_check_log.sum() > 0:
                    for col_imp_log in cols_to_impute_log:
                        if aqi_data_site[col_imp_log].isnull().any(): aqi_data_site[col_imp_log] = aqi_data_site[col_imp_log].ffill().bfill()
                
                aqi_data_site = self._add_lagged_features(aqi_data_site, self.target_names, self.lag_steps)
                aqi_data_site = self._add_rolling_features(aqi_data_site, [c for c in cols_for_rolling if c in aqi_data_site.columns], self.rolling_windows)
                
                # Ensure all columns defined in self.feature_names are present in aqi_data_site
                for fn in self.feature_names:
                    if fn not in aqi_data_site.columns: aqi_data_site[fn] = np.nan 
                
                aqi_data_site.dropna(subset=self.feature_names + self.target_names, inplace=True)
                
                if len(aqi_data_site) < self.sequence_length + 1:
                    # print(f"警告: 城市 '{city_name}' 处理后数据不足 ({len(aqi_data_site)}行)。"); # Commented out to reduce log spam
                    continue
                
                history_source_df_site = df_city_pivoted[['datetime', self.hour_col_name] + self.target_names].copy() 
                for col_hist_imp in self.target_names:
                    if history_source_df_site[col_hist_imp].isnull().any(): history_source_df_site[col_hist_imp] = history_source_df_site[col_hist_imp].ffill().bfill()
                history_source_df_site.dropna(subset=self.target_names, inplace=True)
                buffer_len = max(self.lag_steps, max(self.rolling_windows) if self.rolling_windows else 0, 0)
                history_needed = self.sequence_length + buffer_len
                if not history_source_df_site.empty and len(history_source_df_site) >= history_needed : # Ensure enough history
                     self.initial_history_df_for_future_features_map[site_id_numeric] = history_source_df_site.iloc[-history_needed:].copy()
                
                X_site_df_values = aqi_data_site[self.feature_names].values 
                y_site_df_values = aqi_data_site[self.target_names].values 
                datetime_site_values = aqi_data_site['datetime'].values

                all_site_dfs_processed.append(pd.DataFrame(X_site_df_values, columns=self.feature_names).assign(site_id_numeric_temp=site_id_numeric, y_data_temp=list(y_site_df_values), datetime_temp=datetime_site_values ))


        if not all_site_dfs_processed: raise ValueError("所有文件处理后均无足够数据。")
        self.num_sites = len(self.site_to_id)
        print(f"总共处理了 {self.num_sites} 个不同的站点/城市。")

        combined_processed_df = pd.concat(all_site_dfs_processed, ignore_index=True)
        
        X_all_values = combined_processed_df[self.feature_names].values
        y_all_log_transformed_values = np.array(list(combined_processed_df['y_data_temp'])) 
        site_ids_all_numeric = combined_processed_df['site_id_numeric_temp'].values
        datetime_all_values = combined_processed_df['datetime_temp'].values


        X_all_scaled = self.scaler_X.fit_transform(X_all_values)
        y_all_scaled = self.scaler_y.fit_transform(y_all_log_transformed_values) 

        for site_id_num_map in range(self.num_sites):
            site_mask_map = (site_ids_all_numeric == site_id_num_map)
            site_X_scaled_map = X_all_scaled[site_mask_map]
            if len(site_X_scaled_map) >= self.sequence_length:
                 self.last_sequence_for_future_pred_scaled_map[site_id_num_map] = site_X_scaled_map[-self.sequence_length:]


        X_sequences_final, y_sequences_final = self._create_sequences(X_all_scaled, y_all_scaled, self.sequence_length)
        site_ids_for_sequences_final = site_ids_all_numeric[self.sequence_length : self.sequence_length + len(X_sequences_final)]
        datetime_for_y_sequences_final = datetime_all_values[self.sequence_length : self.sequence_length + len(X_sequences_final)]

        if X_sequences_final.shape[0] == 0: raise ValueError("创建最终序列后没有足够的样本。")
        
        train_size = int(X_sequences_final.shape[0] * 0.8)
        if train_size == 0 or (X_sequences_final.shape[0] - train_size) == 0: raise ValueError(f"最终序列数据量太小。")

        self.X_train = X_sequences_final[:train_size]
        self.X_test = X_sequences_final[train_size:]
        self.y_train = y_sequences_final[:train_size]
        self.y_test = y_sequences_final[train_size:]
        self.site_ids_train = site_ids_for_sequences_final[:train_size]
        self.site_ids_test = site_ids_for_sequences_final[train_size:]
        self.test_dates = pd.Series(datetime_for_y_sequences_final[train_size:])


    def build_model(self, d_model=64, num_transformer_blocks=1, num_heads=4, ff_dim=128, dropout_rate=0.1, learning_rate=0.001, use_cnn_prefix=False):
        sequence_input = Input(shape=(self.sequence_length, len(self.feature_names)), name="sequence_input")
        site_id_input = Input(shape=(1,), name="site_id_input") 

        site_embedding_dim = d_model // 4 
        site_embedding = Embedding(input_dim=self.num_sites, output_dim=site_embedding_dim, name="site_embedding")(site_id_input)
        site_embedding = Flatten()(site_embedding) 
        site_embedding_repeated = keras.layers.RepeatVector(self.sequence_length)(site_embedding)

        x = Dense(d_model - site_embedding_dim)(sequence_input) 
        x = PositionalEncoding(self.sequence_length, d_model - site_embedding_dim)(x) 
        
        x = Concatenate(axis=-1, name="concat_features_site_embedding")([x, site_embedding_repeated])
        x = LayerNormalization(epsilon=1e-6)(x) 
        x = Dropout(dropout_rate)(x) 

        if use_cnn_prefix:
            x = Conv1D(filters=d_model, kernel_size=3, padding='causal', activation='relu')(x) 
            x = LayerNormalization(epsilon=1e-6)(x)

        for _ in range(num_transformer_blocks):
            x = self._transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
        
        x = x[:, -1, :] 
        outputs = Dense(len(self.target_names), kernel_regularizer=keras.regularizers.l2(1e-4))(x) 
        
        model = Model(inputs=[sequence_input, site_id_input], outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0) 
        model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=['mae']) 
        return model

    def train_and_evaluate(self, model, epochs=50, 
                           X_train_seq_override=None, X_train_site_override=None, y_train_override=None, 
                           X_test_seq_override=None, X_test_site_override=None, y_test_override=None):
        
        X_tr_seq = X_train_seq_override if X_train_seq_override is not None else self.X_train
        X_tr_site = X_train_site_override if X_train_site_override is not None else self.site_ids_train
        y_tr = y_train_override if y_train_override is not None else self.y_train
        
        X_te_seq = X_test_seq_override if X_test_seq_override is not None else self.X_test
        X_te_site = X_test_site_override if X_test_site_override is not None else self.site_ids_test
        y_te_scaled = y_test_override if y_test_override is not None else self.y_test 

        print(f"开始在 {epochs} 个周期内训练模型...")
        
        initial_lr = model.optimizer.learning_rate.numpy() if hasattr(model.optimizer, 'learning_rate') else 0.001
        total_steps = (len(X_tr_seq) // 32 + 1) * epochs 
        warmup_steps = int(total_steps * 0.1) 

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            WarmupCosineDecayScheduler(total_steps=total_steps, warmup_steps=warmup_steps, initial_lr=initial_lr, target_lr_min=1e-7, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1) 
        ]

        model.fit([X_tr_seq, X_tr_site.reshape(-1,1)], y_tr, 
                  epochs=epochs, batch_size=32, validation_split=0.2, 
                  verbose=0, callbacks=callbacks)
                  
        print("模型训练完成。正在进行预测...")
        predictions_scaled = model.predict([X_te_seq, X_te_site.reshape(-1,1)]) 

        if np.isnan(predictions_scaled).any():
            print(f"警告: 模型预测结果 (缩放后) 包含 {np.sum(np.isnan(predictions_scaled))} 个NaN值。")
            return np.full_like(y_te_scaled, np.nan), np.expm1(self.scaler_y.inverse_transform(y_te_scaled))
        
        predictions_log_transformed = self.scaler_y.inverse_transform(predictions_scaled)
        y_test_log_transformed = self.scaler_y.inverse_transform(y_te_scaled) 
        predictions_inversed = np.expm1(predictions_log_transformed)
        y_test_inversed = np.expm1(y_test_log_transformed)

        if np.isnan(predictions_inversed).any(): print(f"警告: 模型预测结果 (逆对数变换后) 包含NaN值。")
        if np.isnan(y_test_inversed).any(): print(f"警告: 测试集真实值 (逆对数变换后) 包含NaN值。")
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
            model = load_model(load_path, custom_objects={'PositionalEncoding': PositionalEncoding, 'WarmupCosineDecayScheduler': WarmupCosineDecayScheduler}) 
            if not isinstance(model.input_shape, list) or len(model.input_shape) != 2:
                print("警告: 加载的模型输入形状不是预期的列表（序列+站点ID）。将忽略。")
                return None
            if model.input_shape[0][1] != self.sequence_length or model.input_shape[0][2] != len(self.feature_names):
                print(f"警告: 加载的模型序列输入形状与当前配置不匹配。将忽略。")
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
                use_cnn = hp.Boolean("use_cnn_prefix")
                d_model = hp.Choice('d_model', [64, 128, 192, 256]) 
                num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3, step=1) 
                num_heads = hp.Choice('num_heads', [4, 8]) 
                if d_model % num_heads != 0: 
                    num_heads = 4 if d_model % 4 == 0 else (2 if d_model % 2 == 0 else 1) 
                
                ff_dim_factor = hp.Choice('ff_dim_factor', [2, 4])
                ff_dim = d_model * ff_dim_factor
                dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.05) 
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG', default=5e-4) 
                
                sequence_input_layer = Input(shape=(self.sequence_length, len(self.feature_names)), name="sequence_input_tuner")
                site_id_input_layer = Input(shape=(1,), name="site_id_input_tuner")
                site_embedding_dim_tuner = d_model // 4
                site_embedding = Embedding(input_dim=self.num_sites, output_dim=site_embedding_dim_tuner)(site_id_input_layer)
                site_embedding = Flatten()(site_embedding)
                site_embedding_repeated = keras.layers.RepeatVector(self.sequence_length)(site_embedding)

                x = Dense(d_model - site_embedding_dim_tuner)(sequence_input_layer)
                x = PositionalEncoding(self.sequence_length, d_model - site_embedding_dim_tuner)(x)
                x = Concatenate(axis=-1)([x, site_embedding_repeated]) # Concatenate instead of Add
                x = LayerNormalization(epsilon=1e-6)(x)
                x = Dropout(dropout_rate)(x)

                if use_cnn:
                    cnn_filters = hp.Choice('cnn_filters_tuner', [d_model//2, d_model])
                    cnn_kernels = hp.Choice('cnn_kernel_size_tuner', [3,5])
                    x = Conv1D(filters=cnn_filters, kernel_size=cnn_kernels, padding='causal', activation='relu')(x)
                    x = LayerNormalization(epsilon=1e-6)(x)
                
                for _ in range(num_transformer_blocks):
                    x = self._transformer_encoder_block(x, d_model, num_heads, ff_dim, dropout_rate)
                x = x[:, -1, :] 
                outputs = Dense(len(self.target_names), kernel_regularizer=keras.regularizers.l2(1e-4))(x) 
                model = Model(inputs=[sequence_input_layer, site_id_input_layer], outputs=outputs)
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
                model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=['mae'])
                return model

            tuner = BayesianOptimization(build_tuned_model, objective='val_loss', max_trials=20, executions_per_trial=1, 
                                         directory=tuner_base_dir, project_name='aqi_multi_site_transformer_opt_v2', overwrite=True)
            early_stopping_tuner = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
            
            tuner.search([self.X_train, self.site_ids_train.reshape(-1,1)], self.y_train, 
                         epochs=100, validation_split=0.2, verbose=1, 
                         callbacks=[early_stopping_tuner])
            
            best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hp_list: print("警告: Tuner 未找到最佳超参数。"); return self.build_model()
            best_hp = best_hp_list[0]
            print(f"最佳超参数: use_cnn={best_hp.get('use_cnn_prefix')}, d_model={best_hp.get('d_model')}, blocks={best_hp.get('num_transformer_blocks')}, heads={best_hp.get('num_heads')}, ff_factor={best_hp.get('ff_dim_factor')}, dropout={best_hp.get('dropout')}, lr={best_hp.get('learning_rate')}")
            best_models = tuner.get_best_models(num_models=1)
            if not best_models: print("警告: Tuner 未能构建最佳模型。"); return self.build_model()
            optimized_model = best_models[0]
            if not optimized_model.optimizer:
                optimized_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hp.get('learning_rate')), loss=keras.losses.Huber(), metrics=['mae'])
            return optimized_model
        except Exception as e: print(f"Keras Tuner 调优出错: {e}"); return self.build_model()
        finally:
            if os.path.exists(tuner_base_dir):
                try: shutil.rmtree(tuner_base_dir)
                except Exception as e_clean: print(f"警告: 清理 Tuner 目录失败: {e_clean}")

    def predict_future(self, model, target_city_name, output_dir, num_steps=72):
        if target_city_name not in self.site_to_id:
            print(f"错误: 城市 '{target_city_name}' 未在训练数据中找到，无法为其进行未来预测。")
            return None
        target_site_id_numeric = self.site_to_id[target_city_name]

        if target_site_id_numeric not in self.last_sequence_for_future_pred_scaled_map:
            print(f"错误: 未找到城市 '{target_city_name}' (ID: {target_site_id_numeric}) 的初始预测序列。")
            return None
        current_input_sequence_scaled = self.last_sequence_for_future_pred_scaled_map[target_site_id_numeric].copy()
        
        if target_site_id_numeric not in self.initial_history_df_for_future_features_map:
            print(f"错误: 未找到城市 '{target_city_name}' (ID: {target_site_id_numeric}) 的特征计算历史。")
            return None
        history_df_for_features = self.initial_history_df_for_future_features_map[target_site_id_numeric].copy()

        future_predictions_all_targets_inversed_list = []
        print(f"\n开始为城市 '{target_city_name}' 自回归预测未来 {num_steps} 小时的所有指标...")
        
        last_known_dt_for_pred_start = self.true_absolute_last_datetime_map.get(target_site_id_numeric, self.true_absolute_last_datetime) 
        if last_known_dt_for_pred_start is None: 
            last_known_dt_for_pred_start = pd.Timestamp.now().replace(minute=0,second=0,microsecond=0)
        
        future_timestamps = pd.date_range(start=last_known_dt_for_pred_start + pd.Timedelta(hours=1), periods=num_steps, freq='h')
        print(f"信息: 未来预测将从 {future_timestamps[0]} 开始，直到 {future_timestamps[-1]}。")

        site_id_input_arr = np.array([[target_site_id_numeric]])

        for i in range(num_steps):
            current_input_reshaped_seq = current_input_sequence_scaled.reshape(1, self.sequence_length, len(self.feature_names))
            pred_step_all_targets_scaled = model.predict([current_input_reshaped_seq, site_id_input_arr], verbose=0)[0]
            pred_step_all_targets_log_transformed = self.scaler_y.inverse_transform(pred_step_all_targets_scaled.reshape(1,-1))[0]
            pred_step_all_targets_original_scale = np.expm1(pred_step_all_targets_log_transformed)
            future_predictions_all_targets_inversed_list.append(pred_step_all_targets_original_scale)
            
            new_predicted_target_row_dict = {name: pred_step_all_targets_original_scale[j] for j, name in enumerate(self.target_names)}
            new_predicted_target_row_dict['datetime'] = future_timestamps[i] 
            new_predicted_target_row_dict[self.hour_col_name] = future_timestamps[i].hour
            new_predicted_target_row_dict[self.site_name_col] = target_city_name 
            new_predicted_target_row_dict[self.site_id_col] = target_site_id_numeric
            new_predicted_target_row_df = pd.DataFrame([new_predicted_target_row_dict])
            
            history_df_for_features = pd.concat([history_df_for_features, new_predicted_target_row_df[['datetime', self.hour_col_name] + self.target_names]], ignore_index=True)
            
            max_lookback_needed_for_next_features = self.sequence_length + max(self.lag_steps, max(self.rolling_windows) if self.rolling_windows else 0, 0)
            history_df_for_features = history_df_for_features.iloc[-max_lookback_needed_for_next_features:].copy()

            df_for_next_input_features_raw = history_df_for_features.iloc[-self.sequence_length:].copy()
            df_next_features_engineered = df_for_next_input_features_raw.copy()

            for col in self.target_names:
                if col in df_next_features_engineered.columns:
                    df_next_features_engineered[col] = np.log1p(df_next_features_engineered[col])

            df_next_features_engineered = self._add_time_features(df_next_features_engineered)
            df_next_features_engineered = self._add_lagged_features(df_next_features_engineered, self.target_names, self.lag_steps)
            cols_for_rolling = ['PM2.5', 'PM10', 'O3', 'AQI']
            df_next_features_engineered = self._add_rolling_features(df_next_features_engineered, [c for c in cols_for_rolling if c in df_next_features_engineered.columns], self.rolling_windows)
            
            for fn_check in self.feature_names:
                if fn_check not in df_next_features_engineered.columns:
                    print(f"警告: 在为未来预测步骤 {i+1} 生成特征时，列 '{fn_check}' 缺失。将用NaN填充。")
                    df_next_features_engineered[fn_check] = np.nan

            df_next_features_engineered = df_next_features_engineered[self.feature_names].ffill().bfill() 
            
            next_input_sequence_log_transformed = df_next_features_engineered.values 
            if next_input_sequence_log_transformed.shape[0] != self.sequence_length:
                print(f"错误: 为未来预测步骤 {i+1} 生成的特征序列长度不正确。"); return None 

            current_input_sequence_scaled = self.scaler_X.transform(next_input_sequence_log_transformed)
        
        future_predictions_all_targets_inversed_np = np.array(future_predictions_all_targets_inversed_list)
        df_future_with_timestamp = pd.DataFrame(future_predictions_all_targets_inversed_np, columns=self.target_names) # Renamed for clarity
        df_future_with_timestamp.insert(0, 'Timestamp', future_timestamps)
        
        df_excel = df_future_with_timestamp.copy()
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
            plot_series = df_future_with_timestamp[target_col] if target_col in df_future_with_timestamp.columns else df_excel[target_col] 
            ax.plot(future_timestamps, plot_series, label=f'预测 {target_col}', color='purple', marker='.', ls='-')
            ax.set_ylabel(target_col, fontsize=10); ax.legend(loc='upper right', fontsize=8); ax.grid(True, ls='--', alpha=0.7)
        axes[-1].set_xlabel('时间', fontsize=12); plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')); plt.xticks(rotation=30, ha='right'); plt.tight_layout(rect=[0,0,1,0.98]); plt.show()
        return df_future_with_timestamp 

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
            pred_col_name_in_merged = target_col # df_forecast has original names
            true_col_name_in_merged = target_col + '_true' 
            
            if pred_col_name_in_merged not in merged_df.columns or true_col_name_in_merged not in merged_df.columns:
                print(f"警告: 指标 '{target_col}' 在合并DF中未找到预测列 '{pred_col_name_in_merged}' 或真实列 '{true_col_name_in_merged}'。跳过。")
                continue
            y_pred, y_true = merged_df[pred_col_name_in_merged].astype(float).values, merged_df[true_col_name_in_merged].astype(float).values
            valid = ~np.isnan(y_pred)&~np.isinf(y_pred)&~np.isnan(y_true)&~np.isinf(y_true)
            if not np.any(valid): print(f"警告: 指标 '{target_col}' (未来对比) 无有效数据。"); continue
            y_true_f, y_pred_f = y_true[valid], y_pred[valid]
            if len(y_true_f) == 0: print(f"警告: 指标 '{target_col}' (未来对比) 无有效对。"); continue
            deviations = np.abs(y_pred_f - y_true_f) / (y_true_f + 1e-8)
            perc_exceed = np.sum(deviations > deviation_threshold) / len(y_true_f)
            print(f"信息 (未来对比): '{target_col}' 偏差>阈值百分比: {perc_exceed*100:.2f}% (触发阈值: {points_threshold_percentage*100:.0f}%)")
            if perc_exceed >= points_threshold_percentage: print(f"触发: '{target_col}' 预测性能不佳。"); triggered = True
        return triggered

    def identify_prediction_anomalies(self, model, X_data_scaled, y_data_scaled, dates, site_ids_for_eval, threshold_multiplier=3):
        print("\n--- 开始检测预测异常点 (基于评估测试集) ---")
        if X_data_scaled is None or y_data_scaled is None or dates is None or len(X_data_scaled) == 0:
            print("警告: 无法执行异常检测，缺少数据。")
            return

        predictions_scaled = model.predict([X_data_scaled, site_ids_for_eval]) # Pass site_ids
        predictions_log_transformed = self.scaler_y.inverse_transform(predictions_scaled)
        y_true_log_transformed = self.scaler_y.inverse_transform(y_data_scaled)
        
        predictions_inversed = np.expm1(predictions_log_transformed)
        y_true_inversed = np.expm1(y_true_log_transformed)

        anomaly_reports = []
        for i, target_name in enumerate(self.target_names):
            true_vals = y_true_inversed[:, i]
            pred_vals = predictions_inversed[:, i]
            errors = np.abs(true_vals - pred_vals)
            valid_errors = errors[~np.isnan(errors)]
            if len(valid_errors) == 0: continue
            mean_error = np.mean(valid_errors) 
            std_error = np.std(valid_errors)
            threshold = mean_error + threshold_multiplier * std_error
            anomalous_indices = np.where(errors > threshold)[0]
            if len(anomalous_indices) > 0:
                print(f"\n指标 '{target_name}' 的预测异常点 (误差 > {threshold:.2f}):")
                for idx in anomalous_indices:
                    print(f"  时间: {dates.iloc[idx]}, 真实值: {true_vals[idx]:.2f}, 预测值: {pred_vals[idx]:.2f}, 误差: {errors[idx]:.2f}")
            else: print(f"\n指标 '{target_name}': 未检测到显著预测异常点。")


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
def select_folder_path(title="请选择包含CSV/Excel文件的根文件夹"):
    root = tk.Tk(); root.withdraw(); path = filedialog.askdirectory(title=title); root.destroy(); return path
def select_model_file_path(title="选择模型文件路径", default_name="model.keras"):
    root = tk.Tk(); root.withdraw(); path = filedialog.asksaveasfilename(title=title, initialfile=default_name, defaultextension=".keras", filetypes=[("Keras", "*.keras"), ("HDF5", "*.h5"), ("All", "*.*")]); root.destroy(); return path
def select_verification_data_path(title="请选择验证/测试数据集 (例如 南京_AQI_Data_test.xlsx)"):
    root = tk.Tk(); root.withdraw(); path = filedialog.askopenfilename(title=title, filetypes=[("Excel", "*.xlsx"), ("All", "*.*")]); root.destroy(); return path


def main(): 
    root = tk.Tk(); root.withdraw() 
    predict_future_choice = messagebox.askyesno("未来预测", "您想使用模型预测未来3天每小时的所有指标吗？")
    train_model_choice = messagebox.askyesno("模型训练", "您想训练或重新训练预测模型吗？")
    root.destroy() 

    data_root_folder = select_folder_path("请选择包含所有站点CSV/Excel文件的根文件夹")
    if not data_root_folder: print("未选择数据根文件夹，程序退出。"); return

    verification_data_path = select_verification_data_path(
        "请选择一个验证/测试数据集 (例如 南京_AQI_Data_test.xlsx)\n此文件将用于确定城市名、预测输出目录，并作为未来预测的真实值参考。"
    )
    city_name_for_forecast, forecast_output_dir = "DefaultCity", "." 
    if not verification_data_path:
        print("警告: 未选择验证/测试数据集。未来预测文件名和路径将使用默认值，且无法进行基于未来预测的自动重训练。")
    else:
        forecast_output_dir = os.path.dirname(verification_data_path)
        base_name = os.path.basename(verification_data_path)
        match = re.match(r"([^_]+)_AQI_Data", base_name, re.UNICODE)
        city_name_for_forecast = match.group(1) if match else os.path.splitext(base_name)[0].split('_')[0]
        print(f"未来预测将使用城市名: '{city_name_for_forecast}' 并保存在目录: '{forecast_output_dir}'")


    model_path = select_model_file_path("指定模型文件的加载/保存路径", f"{city_name_for_forecast}_multi_site_transformer_model.keras")
    if not model_path: print("未指定模型文件路径，程序退出。"); return
    
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        try: os.makedirs(model_dir); print(f"已创建模型目录: {model_dir}")
        except OSError as e: print(f"错误: 创建模型目录 {model_dir} 失败: {e}。"); model_path = os.path.basename(model_path)

    try: predictor = AQIPredictor(data_root_folder, model_path) 
    except Exception as e: print(f"初始化AQIPredictor失败: {e}"); return

    model_to_use, y_test_eval_final, initial_preds_on_eval_set, final_preds_on_eval_set = None, None, None, None
    max_retrain_attempts, current_attempt = 2, 0 
    run_user_optimization = False 
    
    static_target_names = predictor.target_names
    eval_template = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}
    eval_initial_model = {target: eval_template.copy() for target in static_target_names} 
    eval_final_model = {target: eval_template.copy() for target in static_target_names} 

    X_eval_set, y_eval_set_scaled, dates_eval_set, site_ids_eval_set = predictor.X_test, predictor.y_test, predictor.test_dates, predictor.site_ids_test
    if X_eval_set is None or y_eval_set_scaled is None or X_eval_set.size == 0:
        messagebox.showerror("错误", "没有可用的内部测试数据进行模型评估。程序将退出。")
        return
    y_test_eval_final = np.expm1(predictor.scaler_y.inverse_transform(y_eval_set_scaled))
    predictor.test_dates = dates_eval_set 


    if train_model_choice:
        model_to_use = predictor.load_existing_model()
        if model_to_use:
            print("重新训练已加载的基线模型 (微调)...")
            opt = keras.optimizers.Adam(learning_rate=1e-5); model_to_use.compile(optimizer=opt, loss=keras.losses.Huber(), metrics=['mae'])
            initial_preds_on_eval_set, _ = predictor.train_and_evaluate(model_to_use, epochs=30, X_test_override=X_eval_set, X_test_site_override=site_ids_eval_set.reshape(-1,1), y_test_override=y_eval_set_scaled)
        else:
            print("构建并训练新的基线模型...")
            model_to_use = predictor.build_model(d_model=128, num_transformer_blocks=2, num_heads=4, ff_dim=256, learning_rate=0.001)
            initial_preds_on_eval_set, _ = predictor.train_and_evaluate(model_to_use, epochs=150, X_test_override=X_eval_set, X_test_site_override=site_ids_eval_set.reshape(-1,1), y_test_override=y_eval_set_scaled)
        predictor.save_model(model_to_use)
    else:
        model_to_use = predictor.load_existing_model()
        if not model_to_use: messagebox.showerror("错误", "未找到模型且未选择训练。"); return
        if not model_to_use.optimizer: model_to_use.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.Huber(), metrics=['mae'])
        preds_scaled = model_to_use.predict([X_eval_set, site_ids_eval_set.reshape(-1,1)])
        preds_log_transformed = predictor.scaler_y.inverse_transform(preds_scaled)
        initial_preds_on_eval_set = np.expm1(preds_log_transformed)


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
                    previous_best_model_rmse = eval_final_model.get('AQI', {}).get('RMSE', float('inf'))
                    final_model_for_future_prediction = tuned_model_candidate 
                    print("训练新调优的模型...")
                    current_optimized_preds, _ = predictor.train_and_evaluate(final_model_for_future_prediction, epochs=150, X_test_override=X_eval_set, X_test_site_override=site_ids_eval_set.reshape(-1,1), y_test_override=y_eval_set_scaled)
                    predictor.save_model(final_model_for_future_prediction) 
                    if y_test_eval_final is not None and current_optimized_preds is not None:
                         eval_newly_optimized = predictor.evaluate(y_test_eval_final, current_optimized_preds)
                         new_rmse = eval_newly_optimized.get('AQI', {}).get('RMSE', float('inf'))
                         if pd.notna(new_rmse) and new_rmse < previous_best_model_rmse :
                             print(f"新优化模型性能提升 (AQI RMSE: {new_rmse:.2f} < {previous_best_model_rmse:.2f})。采用新模型。")
                             eval_final_model = eval_newly_optimized
                             final_preds_on_eval_set_for_plot = current_optimized_preds 
                             optimization_attempted_and_succeeded = True
                         else:
                             print(f"新优化模型性能未提升 (AQI RMSE: {new_rmse:.2f} >= {previous_best_model_rmse:.2f})。回滚到之前的模型。")
                             final_model_for_future_prediction = model_to_use # Rollback to model before this tune attempt
                             optimization_attempted_and_succeeded = False 
                    else: 
                        final_model_for_future_prediction = model_to_use 
                        optimization_attempted_and_succeeded = False
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
            previous_best_model_user = final_model_for_future_prediction 
            previous_best_eval_user = eval_final_model.copy()
            tuned_model_candidate = predictor.tune_model()
            if tuned_model_candidate:
                final_model_for_future_prediction = tuned_model_candidate
                print("训练和评估用户选择优化后的模型...")
                final_preds_on_eval_set_for_plot, _ = predictor.train_and_evaluate(final_model_for_future_prediction, epochs=150, X_test_override=X_eval_set, X_test_site_override=site_ids_eval_set.reshape(-1,1), y_test_override=y_eval_set_scaled)
                predictor.save_model(final_model_for_future_prediction)
                if y_test_eval_final is not None and final_preds_on_eval_set_for_plot is not None:
                    eval_newly_optimized_user = predictor.evaluate(y_test_eval_final, final_preds_on_eval_set_for_plot)
                    rmse_new_user = eval_newly_optimized_user.get('AQI', {}).get('RMSE', float('inf'))
                    rmse_old_user = previous_best_eval_user.get('AQI', {}).get('RMSE', float('inf'))
                    if pd.notna(rmse_new_user) and pd.notna(rmse_old_user) and rmse_new_user < rmse_old_user:
                        print(f"用户优化模型性能提升 (AQI RMSE: {rmse_new_user:.2f} < {rmse_old_user:.2f})。采用新模型。")
                        eval_final_model = eval_newly_optimized_user
                        optimization_attempted_and_succeeded = True
                    else:
                        print(f"用户优化模型性能未提升 (AQI RMSE: {rmse_new_user:.2f} >= {rmse_old_user:.2f})。回滚。")
                        final_model_for_future_prediction = previous_best_model_user
                        optimization_attempted_and_succeeded = False
            else: print("用户选择的优化失败。")

    if final_model_for_future_prediction and X_eval_set is not None and y_eval_set_scaled is not None and dates_eval_set is not None:
        print("\n--- 在评估测试集上进行预测异常点检测 (使用最终模型) ---")
        site_ids_for_anomaly = site_ids_eval_set.reshape(-1,1) if X_eval_set is predictor.X_test else np.zeros((X_eval_set.shape[0],1), dtype=int)
        predictor.identify_prediction_anomalies(final_model_for_future_prediction, X_eval_set, y_eval_set_scaled, dates_eval_set, site_ids_for_anomaly=site_ids_for_anomaly)


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
            available_sites = list(predictor.site_to_id.keys())
            if not available_sites:
                messagebox.showerror("错误", "没有可用于预测的站点信息。")
                return

            root_pred_city = tk.Tk(); root_pred_city.withdraw()
            city_to_predict_for = messagebox.simpledialog.askstring( 
                "指定预测城市", 
                f"请输入要进行未来预测的城市名称。\n可用城市/站点包括: {', '.join(available_sites[:5])}{'...' if len(available_sites)>5 else ''}\n(如果城市名包含空格，请确保输入正确)",
                parent=None 
            )
            root_pred_city.destroy()

            if city_to_predict_for and city_to_predict_for in predictor.site_to_id:
                predictor.predict_future(final_model_for_future_prediction, city_name=city_to_predict_for, output_dir=forecast_output_dir, num_steps=72)
            elif city_to_predict_for:
                 messagebox.showwarning("未来预测", f"城市 '{city_to_predict_for}' 未在训练数据中找到。无法进行预测。")
            else:
                 messagebox.showinfo("未来预测", "未指定预测城市，跳过未来预测。")
        else: 
            messagebox.showwarning("未来预测", "没有可用的模型进行未来预测。")
            print("没有可用的模型进行未来预测。")

if __name__ == "__main__":
    main()
