import pandas as pd  # 导入pandas库，用于数据处理和操作
import numpy as np  # 导入numpy库，用于数值计算
from sklearn.preprocessing import MinMaxScaler  # 从sklearn库导入MinMaxScaler类，用于数据归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error  # 导入评价指标函数
from keras import Input, Model  # 从keras库导入Input和Model，用于构建神经网络模型
from keras.layers import LSTM, Dense, Dropout, Bidirectional, MultiHeadAttention, LayerNormalization, Flatten # 更新导入
from keras_tuner import BayesianOptimization  # 从keras_tuner库导入贝叶斯优化模块
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
import os  # 导入os库，用于文件操作
from keras.models import load_model  # 导入keras中的load_model函数，用于加载已保存的模型
import tkinter as tk # 导入tkinter库，用于GUI
from tkinter import filedialog, messagebox # 从tkinter导入filedialog和messagebox模块
import tempfile # 导入tempfile模块，用于创建临时文件夹
import shutil # 导入shutil模块，用于删除文件夹
from tensorflow import keras # Import keras from tensorflow
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


# 定义AQI预测模型类
class AQIPredictor:
    def __init__(self, data_path, model_path="trained_model.h5", sequence_length=24): # 添加 sequence_length
        """
        初始化AQI预测器类，进行数据加载和预处理。
        模型将预测所有15个指标。
        """
        self.data_path = data_path
        self.model_path = model_path
        self.sequence_length = sequence_length # 输入序列的长度 (例如，过去24小时)
        
        # 输入特征是14种基本污染物
        self.feature_names = [
            'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
            'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 
            'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 
            'CO', 'CO_24h'
        ]
        # 目标是所有15个指标 (14种污染物 + AQI)
        self.target_names = self.feature_names + ['AQI']
        if 'AQI' in self.feature_names: 
            self.target_names = self.feature_names
        else:
            temp_targets = self.feature_names[:] 
            if 'AQI' not in temp_targets:
                temp_targets.append('AQI')
            self.target_names = temp_targets

        self.scaler_X = MinMaxScaler() 
        self.scaler_y = MinMaxScaler() 
        
        self.date_col_name = 'date' 
        self.hour_col_name = 'hour' 
        self.true_absolute_last_datetime = None
        self.last_sequence_for_future_pred_scaled = None # 存储最后一个用于启动未来预测的X序列（已缩放）
        
        self._load_and_prepare_data() 
    
    def _create_sequences(self, X_data, y_data, sequence_length):
        """辅助函数，用于从数据创建输入序列和对应目标。"""
        Xs, ys = [], []
        for i in range(len(X_data) - sequence_length):
            Xs.append(X_data[i:(i + sequence_length)]) # X是前sequence_length个时间步的特征
            ys.append(y_data[i + sequence_length])    # y是sequence_length之后那个时间步的目标
        return np.array(Xs), np.array(ys)

    def _load_and_prepare_data(self, file_path=None, is_external_test_set=False):
        """
        加载并预处理数据，创建序列数据。
        如果 is_external_test_set 为 True，则仅进行转换，不重新拟合scaler。
        返回: X_reshaped, y_scaled, datetime_series (如果 is_external_test_set is True)
               None, None, None (如果 is_external_test_set is False, 数据存储在实例变量中)
        """
        path_to_load = file_path if file_path else self.data_path
        try:
            aqi_data_original = pd.read_excel(path_to_load)
        except FileNotFoundError:
            print(f"错误: 数据文件 {path_to_load} 未找到。请检查路径。")
            raise
        except Exception as e:
            print(f"错误: 读取数据文件 {path_to_load} 时发生错误: {e}")
            raise

        if aqi_data_original.empty:
            raise ValueError(f"错误: 读取的数据文件 {path_to_load} 为空。")

        original_cols = aqi_data_original.columns
        missing_all_metrics = [col for col in self.target_names if col not in original_cols]
        if missing_all_metrics:
            raise ValueError(f"数据文件 {path_to_load} 中缺少必要的指标列: {', '.join(missing_all_metrics)}")
        if self.date_col_name not in original_cols or self.hour_col_name not in original_cols:
            raise ValueError(f"数据文件 {path_to_load} 中缺少日期、小时列。")

        try:
            aqi_data_original['datetime_str_orig'] = aqi_data_original[self.date_col_name].astype(str) + \
                                                     aqi_data_original[self.hour_col_name].astype(str).str.zfill(2)
            aqi_data_original['datetime'] = pd.to_datetime(aqi_data_original['datetime_str_orig'], format='%Y%m%d%H')
            aqi_data_original.sort_values(by='datetime', inplace=True)
            if not is_external_test_set:
                self.true_absolute_last_datetime = aqi_data_original['datetime'].iloc[-1]
                print(f"信息: 主数据集中的真实最后时间戳为: {self.true_absolute_last_datetime}")
        except Exception as e:
            print(f"错误: 在数据文件 {path_to_load} 中转换日期和小时列为datetime对象时失败: {e}")
            raise

        aqi_data = aqi_data_original.copy()
        cols_to_impute = self.target_names 
        nan_counts_before = aqi_data[cols_to_impute].isnull().sum()
        
        if nan_counts_before.sum() > 0:
            print(f"信息: 文件 {path_to_load} - 在插补前，指标列中存在以下数量的NaN值:\n{nan_counts_before[nan_counts_before > 0]}")
            for col in cols_to_impute:
                if aqi_data[col].isnull().any(): aqi_data[col] = aqi_data[col].ffill().bfill()
            nan_counts_after = aqi_data[cols_to_impute].isnull().sum()
            if nan_counts_after.sum() > 0:
                print(f"警告: 文件 {path_to_load} - 插补后仍然存在NaN值:\n{nan_counts_after[nan_counts_after > 0]}")
                aqi_data.dropna(subset=cols_to_impute, inplace=True)
                print("信息: 已移除插补后仍包含NaN的行。")
            else: print(f"信息: 文件 {path_to_load} - 所有NaN值已通过ffill().bfill()成功插补。")
        else: print(f"信息: 文件 {path_to_load} - 指标列中没有检测到NaN值。")
        
        if len(aqi_data) < self.sequence_length + 1: # 需要至少 sequence_length + 1 条数据来创建一个 (X,y) 对
            raise ValueError(f"数据文件 {path_to_load} 在插补NaN值后数据过少 (少于 {self.sequence_length + 1} 行)，无法创建序列。")
        
        # 特征 (X) 是14种污染物，目标 (y) 是所有15个指标
        X_df_values = aqi_data[self.feature_names].values
        y_df_values = aqi_data[self.target_names].values
        datetime_values = aqi_data['datetime'].values

        if not is_external_test_set:
            # 对主数据进行fit_transform
            X_scaled = self.scaler_X.fit_transform(X_df_values)
            y_scaled = self.scaler_y.fit_transform(y_df_values)
            
            # 保存用于启动未来预测的最后一个序列 (来自完整插补后的数据)
            self.last_sequence_for_future_pred_scaled = X_scaled[-self.sequence_length:]
            self.last_datetime_for_autoregression_start = datetime_values[-1] # 对应 self.last_sequence_for_future_pred_scaled 的最后一个时间点
            print(f"信息: 主数据集中用于启动自回归预测的最后一个序列的结束时间点为: {self.last_datetime_for_autoregression_start}")

            # 创建训练/测试用的序列数据
            # X_sequences: (samples, sequence_length, num_features=14)
            # y_sequences: (samples, num_targets=15)
            X_sequences, y_sequences = self._create_sequences(X_scaled, y_scaled, self.sequence_length)
            # 对应 y_sequences 的日期时间戳
            datetime_for_y_sequences = datetime_values[self.sequence_length:]


            if X_sequences.shape[0] == 0:
                raise ValueError("创建序列后没有足够的样本进行训练/测试划分。")

            train_size = int(X_sequences.shape[0] * 0.8)
            if train_size == 0 or (X_sequences.shape[0] - train_size) == 0:
                 raise ValueError(f"序列数据量太小 (总共 {X_sequences.shape[0]} 序列)，无法进行有效的训练集和测试集划分。")

            self.X_train = X_sequences[:train_size]
            self.X_test = X_sequences[train_size:]
            self.y_train = y_sequences[:train_size]
            self.y_test = y_sequences[train_size:]
            
            self.test_dates = pd.Series(datetime_for_y_sequences[train_size:])
            return None, None, None 
        else: # 为外部测试集加载和转换数据
            X_ext_scaled = self.scaler_X.transform(X_df_values)
            y_ext_scaled = self.scaler_y.transform(y_df_values)
            
            X_ext_sequences, y_ext_sequences = self._create_sequences(X_ext_scaled, y_ext_scaled, self.sequence_length)
            datetime_for_y_ext_sequences = datetime_values[self.sequence_length:]
            
            if X_ext_sequences.shape[0] == 0:
                print(f"警告: 外部测试文件 {path_to_load} 创建序列后没有足够的样本。")
                return None, None, None

            return X_ext_sequences, y_ext_sequences, pd.Series(datetime_for_y_ext_sequences)


    def _transformer_encoder_block(self, inputs, d_model, num_heads, ff_dim, dropout_rate):
        """Helper function to create a single Transformer Encoder block."""
        # Multi-Head Self-Attention: Q, V, K are all 'inputs'
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )(inputs, inputs, inputs) 
        out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
        ffn_output = Dense(ff_dim, activation="relu")(out1)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output) 
        out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        return out2

    def build_model(self, d_model=64, num_transformer_blocks=1, num_heads=4, ff_dim=128, dropout_rate=0.1, learning_rate=0.001):
        """
        创建并编译基于Transformer Encoder的模型，支持多目标输出。
        输入形状现在是 (sequence_length, num_input_features)
        """
        inputs = Input(shape=(self.sequence_length, len(self.feature_names))) 
        
        x = Dense(d_model, activation="relu")(inputs) 
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
        """
        训练模型并评估在测试集上的表现。
        """
        X_tr = X_train_override if X_train_override is not None else self.X_train
        y_tr = y_train_override if y_train_override is not None else self.y_train
        X_te = X_test_override if X_test_override is not None else self.X_test
        y_te_scaled = y_test_override if y_test_override is not None else self.y_test 

        print(f"开始在 {epochs} 个周期内训练模型...")
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1 
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=7, min_lr=1e-7, verbose=1
        )

        model.fit(X_tr, y_tr, 
                  epochs=epochs, batch_size=32, validation_split=0.2, 
                  verbose=0, callbacks=[early_stopping, reduce_lr]) 
                  
        print("模型训练完成。正在进行预测...")
        predictions_scaled = model.predict(X_te) 

        if np.isnan(predictions_scaled).any():
            nan_count = np.sum(np.isnan(predictions_scaled))
            print(f"警告: 模型预测结果 (缩放后) 包含 {nan_count} 个NaN值。")
            return np.full_like(y_te_scaled, np.nan), self.scaler_y.inverse_transform(y_te_scaled)
        
        predictions_inversed = self.scaler_y.inverse_transform(predictions_scaled)
        y_test_inversed = self.scaler_y.inverse_transform(y_te_scaled) 

        if np.isnan(predictions_inversed).any(): print(f"警告: 模型预测结果 (逆缩放后) 包含NaN值。")
        if np.isnan(y_test_inversed).any(): print(f"警告: 测试集真实值 (逆缩放后) 包含NaN值。")

        return predictions_inversed, y_test_inversed

    def save_model(self, model):
        """保存训练好的模型。"""
        try:
            save_path = self.model_path
            if not save_path.endswith(".keras"):
                save_path = os.path.splitext(save_path)[0] + ".keras"
                print(f"信息: 模型将以 .keras 格式保存到: {save_path}")
            
            model.save(save_path) 
            print(f"模型已保存为 {save_path}")
        except Exception as e:
            print(f"错误: 保存模型失败: {e}")

    def load_existing_model(self):
        """加载已保存的模型，并检查其输入输出维度是否兼容。"""
        load_path = self.model_path
        
        if not os.path.exists(load_path) and load_path.endswith(".h5"):
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                print(f"信息: 未找到 {load_path} (HDF5)，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                load_path = keras_format_path
        elif not os.path.exists(load_path) and not load_path.endswith(".keras"):
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                 print(f"信息: 未找到 {load_path}，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                 load_path = keras_format_path
            elif os.path.exists(self.model_path): 
                load_path = self.model_path
            else: 
                print(f"模型文件 {self.model_path} (或 .keras 变体) 未找到。")
                return None
        elif not os.path.exists(load_path): 
             print(f"模型文件 {load_path} 未找到。")
             return None

        print(f"加载已保存的模型：{load_path}")
        try:
            model = load_model(load_path) 
            
            # 检查输入形状
            # model.input_shape is (None, sequence_length, num_features)
            loaded_seq_len = model.input_shape[1]
            loaded_num_features = model.input_shape[2]
            current_seq_len = self.sequence_length
            current_num_features = len(self.feature_names)

            if loaded_seq_len != current_seq_len or loaded_num_features != current_num_features:
                print(f"警告: 加载的模型输入形状 (序列长度: {loaded_seq_len}, 特征数: {loaded_num_features}) "
                      f"与当前配置 (序列长度: {current_seq_len}, 特征数: {current_num_features}) 不匹配。")
                print("该模型与当前的数据准备设置不兼容。将忽略此加载的模型。")
                return None

            # 检查输出形状
            if model.output_shape[-1] != len(self.target_names):
                print(f"警告: 加载的模型输出维度 ({model.output_shape[-1]}) 与当前期望的目标数量 ({len(self.target_names)}) 不匹配。")
                print("该模型与当前的多目标设置不兼容。将忽略此加载的模型。")
                return None 
                
            print("已加载模型与当前输入输出形状兼容。")
            return model 
        except Exception as e:
            print(f"错误: 加载模型 {load_path} 失败: {e}。")
            return None

    def tune_model(self):
        """使用贝叶斯优化来调节模型超参数。"""
        tuner_base_dir = tempfile.mkdtemp(prefix="aqi_tuner_")
        print(f"Keras Tuner 将使用临时目录: {tuner_base_dir}")
        try:
            def build_tuned_model(hp):
                d_model = hp.Choice('d_model', [32, 64, 128, 192]) 
                num_transformer_blocks = hp.Int('num_transformer_blocks', min_value=1, max_value=3, step=1) 
                num_heads = hp.Choice('num_heads', [2, 4]) # 确保 d_model 能被 num_heads 整除
                if d_model % num_heads != 0: # 简单调整，确保兼容性
                    if d_model == 32 and num_heads > 4: num_heads = 4
                    elif d_model == 64 and num_heads > 8: num_heads = 8 # d_model=64, num_heads=8 is fine
                    elif d_model == 128 and num_heads > 8: num_heads = 8
                    elif d_model == 192 and num_heads > 8: num_heads = 8 # 192/8 = 24
                    # Ensure num_heads is a factor of d_model
                    while d_model % num_heads != 0 and num_heads > 1:
                        num_heads //=2
                    if num_heads == 1 and d_model % 2 == 0: num_heads = 2 # Prefer at least 2 heads if possible
                    elif num_heads == 1 and d_model % 1 == 0: pass # num_heads can be 1
                    else: # Fallback if no easy factor found
                        if d_model >=64: num_heads = 4
                        else: num_heads = 2


                ff_dim_factor = hp.Choice('ff_dim_factor', [2, 4])
                ff_dim = d_model * ff_dim_factor
                dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.05) 
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='LOG', default=5e-4) 

                model = self.build_model(d_model=d_model, 
                                         num_transformer_blocks=num_transformer_blocks,
                                         num_heads=num_heads,
                                         ff_dim=ff_dim,
                                         dropout_rate=dropout_rate, 
                                         learning_rate=learning_rate) 
                return model

            tuner = BayesianOptimization(
                build_tuned_model,
                objective='val_loss',
                max_trials=15, 
                executions_per_trial=1, 
                directory=tuner_base_dir, 
                project_name='aqi_transformer_optimization_v3', 
                overwrite=True 
            )
            early_stopping_tuner = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, verbose=1, restore_best_weights=True 
            )
            tuner.search(self.X_train, self.y_train, 
                         epochs=100, validation_split=0.2, verbose=1, 
                         callbacks=[early_stopping_tuner]) 
            
            best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hp_list:
                print("警告: Keras Tuner 未找到任何最佳超参数。将返回一个使用默认参数构建的新模型。")
                return self.build_model() 
            best_hp = best_hp_list[0]
            print("找到的最佳超参数:")
            print(f"d_model: {best_hp.get('d_model')}")
            print(f"num_transformer_blocks: {best_hp.get('num_transformer_blocks')}")
            print(f"num_heads: {best_hp.get('num_heads')}")
            print(f"ff_dim_factor: {best_hp.get('ff_dim_factor')}")
            print(f"Dropout rate: {best_hp.get('dropout')}")
            print(f"Learning rate: {best_hp.get('learning_rate')}")
            
            best_models = tuner.get_best_models(num_models=1)
            if not best_models:
                print("警告: Keras Tuner 未能构建最佳模型。将返回一个使用默认参数构建的新模型。")
                return self.build_model() 
            optimized_model = best_models[0]
            if not optimized_model.optimizer:
                print("警告: 从Tuner获取的最佳模型未编译。正在使用找到的最佳超参数重新编译...")
                optimized_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hp.get('learning_rate')), 
                                        loss='mean_squared_error', metrics=['mae'])
            return optimized_model
        except Exception as e: 
            print(f"Keras Tuner 调优过程中发生错误: {e}")
            print("将返回一个使用默认参数构建的新模型。")
            return self.build_model() 
        finally:
            if os.path.exists(tuner_base_dir):
                try: shutil.rmtree(tuner_base_dir)
                except Exception as e_clean: print(f"警告: 清理 Keras Tuner 临时目录 {tuner_base_dir} 失败: {e_clean}")

    def predict_future(self, model, city_name, output_dir, num_steps=72):
        """
        使用最后一个已知的特征集预测未来指定步数的所有指标 (自回归方式)。
        并将预测结果保存到Excel文件。
        """
        if self.last_sequence_for_future_pred_scaled is None or self.last_sequence_for_future_pred_scaled.shape[0] == 0:
            print("错误: 没有可用于初始化未来预测的已缩放输入序列 (last_sequence_for_future_pred_scaled)。")
            return None
        
        current_input_sequence_scaled = self.last_sequence_for_future_pred_scaled.copy() # Shape: (sequence_length, num_input_features)
        
        future_predictions_all_targets_scaled_list = []
        print(f"\n开始自回归预测未来 {num_steps} 小时的所有指标...")
        
        start_prediction_datetime = self.true_absolute_last_datetime
        if start_prediction_datetime is None:
            print("错误: 未能确定原始数据文件的真实最后一个历史时间戳。")
            start_prediction_datetime = self.last_datetime_for_autoregression_start if self.last_datetime_for_autoregression_start else pd.Timestamp.now().replace(minute=0,second=0,microsecond=0)
            print(f"警告: 将使用 {start_prediction_datetime} 作为未来预测的基准。")

        future_timestamps = pd.date_range(
            start=start_prediction_datetime + pd.Timedelta(hours=1), 
            periods=num_steps, freq='h'
        )
        print(f"信息: 未来预测将从 {future_timestamps[0]} 开始，直到 {future_timestamps[-1]}。")
        print(f"信息: 用于启动自回归预测的最后一个已知特征序列的结束时间戳是 {self.last_datetime_for_autoregression_start}。")

        for i in range(num_steps):
            current_input_reshaped = current_input_sequence_scaled.reshape(1, self.sequence_length, len(self.feature_names))
            prediction_step_all_targets_scaled = model.predict(current_input_reshaped, verbose=0)[0] 
            future_predictions_all_targets_scaled_list.append(prediction_step_all_targets_scaled)
            
            new_feature_vector_scaled = np.zeros(len(self.feature_names))
            for k, feature_name_in_X in enumerate(self.feature_names):
                try:
                    target_idx = self.target_names.index(feature_name_in_X)
                    new_feature_vector_scaled[k] = prediction_step_all_targets_scaled[target_idx]
                except ValueError: 
                    print(f"严重错误: 特征 '{feature_name_in_X}' 未在目标列表 '{self.target_names}' 中找到。自回归失败。")
                    return None 
            
            current_input_sequence_scaled = np.roll(current_input_sequence_scaled, -1, axis=0)
            current_input_sequence_scaled[-1, :] = new_feature_vector_scaled
        
        future_predictions_all_targets_scaled_np = np.array(future_predictions_all_targets_scaled_list) 
        future_predictions_all_targets_inversed = self.scaler_y.inverse_transform(future_predictions_all_targets_scaled_np)
        
        df_future_with_timestamp = pd.DataFrame(future_predictions_all_targets_inversed, columns=self.target_names)
        df_future_with_timestamp.insert(0, 'Timestamp', future_timestamps) 
        
        df_excel = df_future_with_timestamp.copy()
        for col in self.target_names: 
            df_excel[col] = df_excel[col].round().astype(int)

        df_excel['date'] = df_excel['Timestamp'].dt.strftime('%Y%m%d')
        df_excel['hour'] = df_excel['Timestamp'].dt.hour
        
        excel_columns_ordered = ['date', 'hour'] + self.target_names
        df_excel_final = df_excel[excel_columns_ordered]

        print("\n未来3天每小时所有指标预测 (自回归):")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000): 
            print(df_excel_final) 
        
        forecast_file_name = os.path.join(output_dir, f"{city_name}_AQI_Data_forecast.xlsx")
        try:
            df_excel_final.to_excel(forecast_file_name, index=False)
            print(f"未来预测结果已保存到: {forecast_file_name}")
        except Exception as e:
            print(f"错误: 保存未来预测结果到Excel文件失败: {e}")

        num_targets_to_plot = len(self.target_names)
        fig, axes = plt.subplots(num_targets_to_plot, 1, figsize=(12, 3 * num_targets_to_plot), sharex=True)
        if num_targets_to_plot == 1: axes = [axes] 
        
        fig.suptitle('未来3天每小时各项指标预测 (自回归)', fontsize=16, y=1.02)
        for i, target_col in enumerate(self.target_names):
            ax = axes[i]
            ax.plot(df_future_with_timestamp['Timestamp'], df_future_with_timestamp[target_col], label=f'预测 {target_col}', color='purple', marker='.', linestyle='-') 
            ax.set_ylabel(target_col, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        axes[-1].set_xlabel('时间', fontsize=12) 
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(rect=[0, 0, 1, 0.98]) 
        plt.show()
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
        """
        比较未来预测和对应的真实值，并决定是否触发重训练。
        df_forecast 应包含 'Timestamp' 和所有 target_names 列 (原始浮点值)。
        """
        if df_forecast is None or df_forecast.empty:
            print("警告 (compare): 预测DataFrame为空，无法进行比较。")
            return False 
        
        if 'Timestamp' not in df_forecast.columns:
            if 'Timestamp' in df_forecast.index.names:
                print("信息 (compare): 'Timestamp' 在 df_forecast 的索引中找到，将重置索引。")
                df_forecast = df_forecast.reset_index()
            else:
                print(f"错误 (compare): 'Timestamp' 列未在传入的 df_forecast 中找到，也不在其索引中。列: {df_forecast.columns.tolist()}")
                return False 

        if not future_ground_truth_path or not os.path.exists(future_ground_truth_path):
            print(f"警告: 未提供或未找到未来真实值文件: {future_ground_truth_path}。无法比较预测。")
            return False 

        try:
            df_truth_raw = pd.read_excel(future_ground_truth_path)
            if self.date_col_name not in df_truth_raw.columns or self.hour_col_name not in df_truth_raw.columns:
                print(f"错误: 未来真实值文件 {future_ground_truth_path} 缺少 'date' 或 'hour' 列。")
                return False
            df_truth_raw['datetime_str'] = df_truth_raw[self.date_col_name].astype(str) + df_truth_raw[self.hour_col_name].astype(str).str.zfill(2)
            df_truth_raw['Timestamp'] = pd.to_datetime(df_truth_raw['datetime_str'], format='%Y%m%d%H')
            
            cols_for_truth_merge = ['Timestamp'] + self.target_names
            missing_cols_in_truth = [col for col in cols_for_truth_merge if col not in df_truth_raw.columns]
            if missing_cols_in_truth:
                print(f"错误: 未来真实值文件 {future_ground_truth_path} 缺少用于比较的列: {missing_cols_in_truth}")
                return False
            df_truth = df_truth_raw[cols_for_truth_merge].copy()

        except Exception as e:
            print(f"错误: 读取或处理未来真实值文件 {future_ground_truth_path} 失败: {e}")
            return False

        print(f"DEBUG (compare): df_forecast columns for merge: {df_forecast.columns.tolist()}")
        print(f"DEBUG (compare): df_truth columns for merge: {df_truth.columns.tolist()}")
        
        merged_df = pd.merge(df_forecast, df_truth, on='Timestamp', suffixes=('_pred', '_true'), how='inner')

        if merged_df.empty:
            print("警告: 预测与真实值文件在Timestamp上没有重叠数据，无法比较。")
            return False

        triggered_for_retraining = False
        for target_col_name in self.target_names:
            pred_col_name_in_merged = target_col_name # df_forecast 中的列名 (在merge前没有后缀)
            true_col_name_in_merged = target_col_name + '_true' # df_truth 中的列名 (在merge后加了_true后缀)
            
            # 修正：df_forecast 在 merge 前没有 _pred 后缀，所以直接用 target_col_name
            if target_col_name not in merged_df.columns or true_col_name_in_merged not in merged_df.columns:
                # 如果是 df_forecast 的列在 merge 后应该有 _pred 后缀
                pred_col_name_in_merged = target_col_name + '_pred'
                if pred_col_name_in_merged not in merged_df.columns or true_col_name_in_merged not in merged_df.columns:
                    print(f"警告: 指标 '{target_col_name}' 在合并后的DataFrame中未找到预测列 '{pred_col_name_in_merged}' 或真实列 '{true_col_name_in_merged}'。跳过此指标。")
                    continue

            y_pred_f = merged_df[pred_col_name_in_merged].values.astype(float) 
            y_true_f = merged_df[true_col_name_in_merged].values.astype(float)
            
            valid_mask = ~np.isnan(y_pred_f) & ~np.isinf(y_pred_f) & \
                         ~np.isnan(y_true_f) & ~np.isinf(y_true_f)
            
            if not np.any(valid_mask):
                print(f"警告: 指标 '{target_col_name}' (未来对比) 的真实值或预测值全部为NaN/inf。")
                continue
            
            y_true_f_masked = y_true_f[valid_mask]
            y_pred_f_masked = y_pred_f[valid_mask]

            if len(y_true_f_masked) == 0:
                print(f"警告: 指标 '{target_col_name}' (未来对比) 没有有效的真实值和预测值对。")
                continue

            deviations = np.abs(y_pred_f_masked - y_true_f_masked) / (y_true_f_masked + 1e-8) 
            points_exceeding_dev = np.sum(deviations > deviation_threshold)
            percentage_exceeding = points_exceeding_dev / len(y_true_f_masked)

            print(f"信息 (未来对比): 指标 '{target_col_name}' 偏差超过阈值的点所占百分比: {percentage_exceeding*100:.2f}% (触发阈值: {points_threshold_percentage*100:.0f}%)")

            if percentage_exceeding >= points_threshold_percentage:
                print(f"触发条件满足 (未来对比): 指标 '{target_col_name}' 的预测性能不佳。")
                triggered_for_retraining = True
        
        if triggered_for_retraining:
            print("一个或多个指标的未来预测性能不佳，建议重新训练和优化。")
        else:
            print("所有指标的未来预测性能均在可接受范围内。")
        return triggered_for_retraining


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

    # 验证/测试数据集路径 (用于确定城市名、输出目录，并作为未来预测的真实值参考)
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
    run_user_optimization = False # 初始化
    
    static_target_names = predictor.target_names
    eval_template = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}
    eval_initial_model = {target: eval_template.copy() for target in static_target_names} 
    eval_final_model = {target: eval_template.copy() for target in static_target_names} 

    # --- 确定用于评估的测试集 (来自 predictor 内部划分，或 verification_data_path 如果用户选择) ---
    # 当前实现中，常规评估总是使用 predictor.X_test, predictor.y_test
    # verification_data_path 主要用于未来预测的对比
    X_eval_set, y_eval_set_scaled, dates_eval_set = predictor.X_test, predictor.y_test, predictor.test_dates
    if X_eval_set is None or y_eval_set_scaled is None or X_eval_set.size == 0:
        messagebox.showerror("错误", "没有可用的内部测试数据进行模型评估。程序将退出。")
        return
    y_test_eval_final = predictor.scaler_y.inverse_transform(y_eval_set_scaled) 
    predictor.test_dates = dates_eval_set # Ensure predictor uses these dates for plotting eval results


    # --- 初始模型训练/加载 ---
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
    
    # --- Iterative Loop for Future Prediction Comparison and Retraining ---
    final_model_for_future_prediction = model_to_use 
    eval_final_model = eval_initial_model.copy() 
    final_preds_on_eval_set_for_plot = initial_preds_on_eval_set 
    optimization_attempted_and_succeeded = False 

    if predict_future_choice: 
        for attempt in range(max_retrain_attempts + 1):
            print(f"\n--- 未来预测与评估迭代: {attempt + 1} ---")
            df_future = predictor.predict_future(final_model_for_future_prediction, city_name=city_name_for_forecast, output_dir=forecast_output_dir, num_steps=72)
            if df_future is None:
                messagebox.showerror("错误", "未来预测失败，无法继续迭代。"); break
            
            if not verification_data_path: # 如果没有提供验证文件，则无法比较和迭代
                print("未提供未来真实值验证文件，跳过基于未来预测的重训练触发。")
                break

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
                else:
                    print("超参数调优未能产生新模型，停止迭代。")
                    break
            else: 
                if needs_retrain_based_on_future: print("已达到最大自动重训练尝试次数，使用当前模型。")
                else: print("未来预测准确性可接受。")
                break 
    elif train_model_choice: # 训练了模型，但没选未来预测，仍可提供优化选项
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
            else:
                print("用户选择的优化失败。")


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

    # Future prediction using the absolute final model 
    if predict_future_choice:
        if final_model_for_future_prediction: 
            print("\n--- 使用最终确定的模型进行未来所有指标预测 ---")
            predictor.predict_future(final_model_for_future_prediction, city_name=city_name_for_forecast, output_dir=forecast_output_dir, num_steps=72)
        else: 
            messagebox.showwarning("未来预测", "没有可用的模型进行未来预测。")
            print("没有可用的模型进行未来预测。")


if __name__ == "__main__":
    main()
