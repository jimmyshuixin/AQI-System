import pandas as pd  # 导入pandas库，用于数据处理和操作
import numpy as np  # 导入numpy库，用于数值计算
from sklearn.preprocessing import MinMaxScaler  # 从sklearn库导入MinMaxScaler类，用于数据归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error  # 导入评价指标函数
from keras import Input, Model  # 从keras库导入Input和Model，用于构建神经网络模型
from keras.layers import LSTM, Dense, Dropout, Bidirectional  # 导入LSTM、Dense、Dropout、Bidirectional层，用于神经网络结构
from keras_tuner import BayesianOptimization  # 从keras_tuner库导入贝叶斯优化模块
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
import os  # 导入os库，用于文件操作
from keras.models import load_model  # 导入keras中的load_model函数，用于加载已保存的模型
import tkinter as tk # 导入tkinter库，用于GUI
from tkinter import filedialog, messagebox # 从tkinter导入filedialog和messagebox模块
import tempfile # 导入tempfile模块，用于创建临时文件夹
import shutil # 导入shutil模块，用于删除文件夹
from tensorflow import keras # Import keras from tensorflow

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
    def __init__(self, data_path, model_path="trained_model.h5"):
        """
        初始化AQI预测器类，进行数据加载和预处理。
        模型将预测所有15个指标。
        """
        self.data_path = data_path
        self.model_path = model_path
        
        # 输入特征是14种基本污染物
        self.feature_names = [
            'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
            'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 
            'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 
            'CO', 'CO_24h'
        ]
        # 目标是所有15个指标 (14种污染物 + AQI)
        self.target_names = self.feature_names + ['AQI']
        # 确保目标名称列表中的唯一性 (如果AQI已在feature_names中则不需要添加)
        if 'AQI' in self.feature_names: # Should not happen with current setup
            self.target_names = self.feature_names
        else:
             # 确保 target_names 中的顺序，通常将 AQI 放在特定位置或最后
            temp_targets = self.feature_names[:] # 复制
            if 'AQI' not in temp_targets:
                temp_targets.append('AQI')
            self.target_names = temp_targets


        self.scaler_X = MinMaxScaler() # 用于输入特征 (14种污染物)
        self.scaler_y = MinMaxScaler() # 用于目标输出 (所有15个指标)
        
        self.date_col_name = 'date' 
        self.hour_col_name = 'hour' 
        self.true_absolute_last_datetime = None
        self.last_datetime_for_autoregression_start = None
        self.X_input_for_future_pred_init_scaled = None # 存储最后一个用于启动未来预测的X输入（已缩放）
        
        self._load_and_prepare_data()
    
    def _load_and_prepare_data(self):
        """加载并预处理数据，以支持多目标自回归预测。"""
        try:
            aqi_data_original = pd.read_excel(self.data_path)
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.data_path} 未找到。请检查路径。")
            raise
        except Exception as e:
            print(f"错误: 读取数据文件 {self.data_path} 时发生错误: {e}")
            raise

        if aqi_data_original.empty:
            raise ValueError("错误: 读取的原始数据文件为空。")

        original_cols = aqi_data_original.columns
        # 检查所有目标列（即所有15个指标）是否存在
        missing_all_metrics = [col for col in self.target_names if col not in original_cols]
        if missing_all_metrics:
            print(f"错误: 原始数据文件中缺少以下列: {', '.join(missing_all_metrics)}")
            raise ValueError("原始数据文件中缺少必要的指标列。")
        if self.date_col_name not in original_cols:
            raise ValueError(f"原始数据文件中缺少日期列: {self.date_col_name}。")
        if self.hour_col_name not in original_cols:
            raise ValueError(f"原始数据文件中缺少小时列: {self.hour_col_name}。")

        try:
            aqi_data_original['datetime_str_orig'] = aqi_data_original[self.date_col_name].astype(str) + \
                                                     aqi_data_original[self.hour_col_name].astype(str).str.zfill(2)
            aqi_data_original['datetime'] = pd.to_datetime(aqi_data_original['datetime_str_orig'], format='%Y%m%d%H')
            aqi_data_original.sort_values(by='datetime', inplace=True)
            self.true_absolute_last_datetime = aqi_data_original['datetime'].iloc[-1]
            print(f"信息: 原始数据文件中的真实最后时间戳为: {self.true_absolute_last_datetime}")
        except Exception as e:
            print(f"错误: 在原始数据中转换日期和小时列为datetime对象时失败: {e}")
            raise

        aqi_data = aqi_data_original.copy()
        cols_to_impute = self.target_names # 插补所有15个指标列中的NaN
        nan_counts_before = aqi_data[cols_to_impute].isnull().sum()
        
        if nan_counts_before.sum() > 0:
            print(f"信息: 在插补前，指标列中存在以下数量的NaN值:\n{nan_counts_before[nan_counts_before > 0]}")
            for col in cols_to_impute:
                if aqi_data[col].isnull().any():
                    aqi_data[col] = aqi_data[col].ffill().bfill()
            nan_counts_after = aqi_data[cols_to_impute].isnull().sum()
            if nan_counts_after.sum() > 0:
                print(f"警告: 插补后仍然存在NaN值:\n{nan_counts_after[nan_counts_after > 0]}")
                aqi_data.dropna(subset=cols_to_impute, inplace=True)
                print("信息: 已移除插补后仍包含NaN的行。")
            else:
                print("信息: 所有NaN值已通过ffill().bfill()成功插补。")
        else:
            print("信息: 指标列中没有检测到NaN值。")
        
        if aqi_data.empty:
            raise ValueError("数据在插补NaN值后为空。")
        
        # 用于启动未来预测的最后一个实际观测时间点和对应的特征
        self.last_datetime_for_autoregression_start = aqi_data['datetime'].iloc[-1]
        # X的输入特征是 self.feature_names (14种污染物)
        last_input_features_unscaled = aqi_data[self.feature_names].iloc[[-1]].values
        # 在fit_transform之前，不能单独transform最后一行，所以先fit_transform全部X，然后取最后一行
        
        print(f"信息: 用于启动自回归预测的最后一个实际观测时间点为: {self.last_datetime_for_autoregression_start}")

        # 准备 X (特征 t) 和 y (目标 指标 t+1)
        X_df = aqi_data[self.feature_names] # 14 种污染物作为输入特征
        y_df = aqi_data[self.target_names]  # 所有15个指标作为目标

        X_values = X_df.iloc[:-1].values # 特征在 t 时刻
        y_values = y_df.iloc[1:].values  # 目标在 t+1 时刻
        
        min_len = min(len(X_values), len(y_values))
        X_values = X_values[:min_len]
        y_values = y_values[:min_len]
        
        datetime_for_y = aqi_data['datetime'].iloc[1:min_len+1].reset_index(drop=True)

        self.X_scaled_all_for_training = self.scaler_X.fit_transform(X_values)
        self.y_scaled = self.scaler_y.fit_transform(y_values) # scaler_y 现在拟合15个目标

        # 保存用于未来预测的最后一个X输入（已缩放）
        # 这是来自 X_df 的最后一行，经过 scaler_X 变换
        self.X_input_for_future_pred_init_scaled = self.scaler_X.transform(aqi_data[self.feature_names].iloc[[-1]].values)


        if np.isnan(self.X_scaled_all_for_training).any():
            print("警告: 特征数据 X_scaled_all_for_training 在归一化后包含NaN值。")
        if np.isnan(self.y_scaled).any():
            print("警告: 目标数据 y_scaled 在归一化后包含NaN值。")
        
        train_size = int(len(self.X_scaled_all_for_training) * 0.8)
        if train_size == 0 or (len(self.X_scaled_all_for_training) - train_size) == 0:
             raise ValueError(f"数据量太小 (总共 {len(self.X_scaled_all_for_training)} 可用序列)，无法进行有效的训练集和测试集划分。")

        self.X_train = self.X_scaled_all_for_training[:train_size]
        self.X_test = self.X_scaled_all_for_training[train_size:]
        self.y_train = self.y_scaled[:train_size]
        self.y_test = self.y_scaled[train_size:] # y_test 现在是 (samples, num_targets)
        
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

        self.test_dates = datetime_for_y.iloc[train_size:].reset_index(drop=True)


    def build_model(self, units=50, dropout_rate=0.2, activation_func='tanh', learning_rate=0.001):
        """
        创建并编译双向LSTM模型，支持多目标输出。
        """
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2])) # 输入维度是特征数 (14)
        x = Bidirectional(LSTM(units=units, activation=activation_func))(inputs)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(len(self.target_names))(x) # 输出维度是目标数 (15)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae']) # MAE 作为多输出的平均指标
        return model

    def train_and_evaluate(self, model, epochs=50):
        """
        训练模型并评估在测试集上的表现。
        """
        print(f"开始在 {epochs} 个周期内训练模型...")
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        print("模型训练完成。正在进行预测...")
        predictions_scaled = model.predict(self.X_test) # shape: (num_samples, num_targets)

        if np.isnan(predictions_scaled).any():
            nan_count = np.sum(np.isnan(predictions_scaled))
            print(f"警告: 模型预测结果 (缩放后) 包含 {nan_count} 个NaN值。")
            # 返回NaN数组，以便后续评估函数可以处理
            return np.full_like(self.y_test, np.nan), self.scaler_y.inverse_transform(self.y_test)
        
        predictions_inversed = self.scaler_y.inverse_transform(predictions_scaled)
        y_test_inversed = self.scaler_y.inverse_transform(self.y_test)

        if np.isnan(predictions_inversed).any():
            print(f"警告: 模型预测结果 (逆缩放后) 包含NaN值。")
        if np.isnan(y_test_inversed).any():
            print(f"警告: 测试集真实值 (逆缩放后) 包含NaN值。")

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
        """加载已保存的模型，并检查其输出维度是否兼容。"""
        load_path = self.model_path
        # 优先尝试 .keras 格式
        if not os.path.exists(load_path) and load_path.endswith(".h5"):
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                print(f"信息: 未找到 {load_path} (HDF5)，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                load_path = keras_format_path
        elif not os.path.exists(load_path) and not load_path.endswith(".keras"):
             # 如果路径没有扩展名，或者不是 .keras，也尝试 .keras
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                 print(f"信息: 未找到 {load_path}，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                 load_path = keras_format_path
            elif os.path.exists(self.model_path): # 尝试原始路径
                load_path = self.model_path
            else: # 如果两者都不存在
                print(f"模型文件 {self.model_path} (或 .keras 变体) 未找到。")
                return None
        elif not os.path.exists(load_path): # 路径有 .keras 扩展名但文件不存在
             print(f"模型文件 {load_path} 未找到。")
             return None


        print(f"加载已保存的模型：{load_path}")
        try:
            model = load_model(load_path)
            # 检查加载模型的输出维度是否与当前配置兼容
            if model.output_shape[-1] != len(self.target_names):
                print(f"警告: 加载的模型输出维度 ({model.output_shape[-1]}) 与当前期望的目标数量 ({len(self.target_names)}) 不匹配。")
                print("该模型与当前的多目标设置不兼容。将忽略此加载的模型，并训练一个新模型。")
                return None # 表示模型不兼容
            print("已加载模型与当前目标数量兼容。")
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
                units = hp.Int('units', min_value=32, max_value=256, step=32) 
                dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1) 
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='LOG', default=1e-3) 
                activation_func = 'tanh' 

                model = self.build_model(units, dropout_rate, activation_func=activation_func, learning_rate=learning_rate) 
                return model

            tuner = BayesianOptimization(
                build_tuned_model,
                objective='val_loss',
                max_trials=10, 
                executions_per_trial=1, 
                directory=tuner_base_dir, 
                project_name='aqi_optimization', 
                overwrite=True 
            )
            tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, verbose=1) 
            
            best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hp_list:
                print("警告: Keras Tuner 未找到任何最佳超参数。将返回一个使用默认参数构建的新模型。")
                return self.build_model(activation_func='tanh') 

            best_hp = best_hp_list[0]
            print("找到的最佳超参数:")
            print(f"Units: {best_hp.get('units')}")
            print(f"Dropout rate: {best_hp.get('dropout')}")
            print(f"Learning rate: {best_hp.get('learning_rate')}")
            
            best_models = tuner.get_best_models(num_models=1)
            if not best_models:
                print("警告: Keras Tuner 未能构建最佳模型。将返回一个使用默认参数构建的新模型。")
                return self.build_model(activation_func='tanh') 
            
            optimized_model = best_models[0]
            if not optimized_model.optimizer:
                print("警告: 从Tuner获取的最佳模型未编译。正在重新编译...")
                optimized_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_hp.get('learning_rate')), 
                                        loss='mean_squared_error', metrics=['mae'])
            return optimized_model

        except Exception as e: 
            print(f"Keras Tuner 调优过程中发生错误: {e}")
            print("将返回一个使用默认参数构建的新模型。")
            return self.build_model(activation_func='tanh') 
        finally:
            if os.path.exists(tuner_base_dir):
                try:
                    shutil.rmtree(tuner_base_dir)
                    print(f"已清理 Keras Tuner 临时目录: {tuner_base_dir}")
                except Exception as e_clean:
                    print(f"警告: 清理 Keras Tuner 临时目录 {tuner_base_dir} 失败: {e_clean}")

    def predict_future(self, model, num_steps=72):
        """
        使用最后一个已知的特征集预测未来指定步数的AQI (自回归方式)。
        """
        if self.X_input_for_future_pred_init_scaled is None or self.X_input_for_future_pred_init_scaled.shape[0] == 0:
            print("错误: 没有可用于初始化未来预测的已缩放输入特征 (X_input_for_future_pred_init_scaled)。")
            return None
        
        current_input_X_scaled = self.X_input_for_future_pred_init_scaled.copy() # (1, num_input_features)
        current_input_reshaped = current_input_X_scaled.reshape(1, 1, current_input_X_scaled.shape[1])
        
        future_predictions_all_targets_scaled_list = []
        print(f"\n开始自回归预测未来 {num_steps} 小时的所有指标...")
        
        if self.true_absolute_last_datetime is None:
            print("错误: 未能确定原始数据文件的真实最后一个历史时间戳。")
            start_prediction_datetime = self.last_datetime_for_autoregression_start if self.last_datetime_for_autoregression_start else pd.Timestamp.now().replace(minute=0,second=0,microsecond=0)
            print(f"警告: 将使用 {start_prediction_datetime} 作为未来预测的基准。")
        else:
            start_prediction_datetime = self.true_absolute_last_datetime

        future_timestamps = pd.date_range(
            start=start_prediction_datetime + pd.Timedelta(hours=1), 
            periods=num_steps, 
            freq='h'
        )
        print(f"信息: 未来预测将从 {future_timestamps[0]} 开始，直到 {future_timestamps[-1]}。")
        print(f"信息: 用于启动自回归预测的最后一个已知特征集对应的时间戳是 {self.last_datetime_for_autoregression_start}。")

        for i in range(num_steps):
            # model.predict 输入是 (1, 1, num_input_features)
            # model.predict 输出是 (1, num_output_targets) 
            prediction_step_all_targets_scaled = model.predict(current_input_reshaped, verbose=0)[0] # shape (num_targets,)
            future_predictions_all_targets_scaled_list.append(prediction_step_all_targets_scaled)
            
            # 自回归步骤：用预测的污染物值更新下一个输入
            # 下一个输入 current_input_X_scaled 的特征应该与 self.feature_names (14个污染物) 对应
            next_input_X_scaled_values = np.zeros(len(self.feature_names))
            for k, feature_name_in_X in enumerate(self.feature_names):
                try:
                    # 从预测的目标(prediction_step_all_targets_scaled)中找到对应的污染物值
                    # self.target_names 包含了所有15个指标，包括AQI
                    target_idx = self.target_names.index(feature_name_in_X)
                    next_input_X_scaled_values[k] = prediction_step_all_targets_scaled[target_idx]
                except ValueError:
                    # 这不应该发生，因为 self.feature_names 是 self.target_names 的子集
                    print(f"严重错误: 特征 '{feature_name_in_X}' 未在目标列表 '{self.target_names}' 中找到。自回归失败。")
                    return None 
            
            current_input_X_scaled = next_input_X_scaled_values.reshape(1, -1)
            current_input_reshaped = current_input_X_scaled.reshape(1, 1, current_input_X_scaled.shape[1])
        
        future_predictions_all_targets_scaled_np = np.array(future_predictions_all_targets_scaled_list) # shape (num_steps, num_targets)
        future_predictions_all_targets_inversed = self.scaler_y.inverse_transform(future_predictions_all_targets_scaled_np)
        
        df_future = pd.DataFrame(future_predictions_all_targets_inversed, columns=self.target_names)
        df_future.insert(0, 'Timestamp', future_timestamps)
        
        print("\n未来3天每小时所有指标预测 (自回归):")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000): 
            print(df_future)

        # 绘制所有预测指标的图表
        num_targets_to_plot = len(self.target_names)
        fig, axes = plt.subplots(num_targets_to_plot, 1, figsize=(12, 3 * num_targets_to_plot), sharex=True)
        if num_targets_to_plot == 1: 
            axes = [axes] 
        
        fig.suptitle('未来3天每小时各项指标预测 (自回归)', fontsize=16, y=1.02)
        for i, target_col in enumerate(self.target_names):
            ax = axes[i]
            ax.plot(df_future['Timestamp'], df_future[target_col], label=f'预测 {target_col}', color='purple', marker='.', linestyle='-')
            ax.set_ylabel(target_col, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        axes[-1].set_xlabel('时间', fontsize=12) 
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(rect=[0, 0, 1, 0.98]) 
        plt.show()

        return df_future


    @staticmethod
    def evaluate(y_true_multi, y_pred_multi):
        """
        计算多目标输出的评价指标。
        返回一个字典，键是目标名称，值是该目标的评估指标字典。
        """
        if y_true_multi is None or y_pred_multi is None:
            return {target: {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')} for target in AQIPredictor.static_target_names_for_eval()}

        all_metrics_eval = {}
        num_targets = y_true_multi.shape[1]
        target_names_for_eval = AQIPredictor.static_target_names_for_eval() 

        if num_targets != len(target_names_for_eval):
            print(f"警告: 评估时目标数量 ({num_targets}) 与预期目标名称数量 ({len(target_names_for_eval)}) 不匹配。")
            target_names_to_use = target_names_for_eval[:num_targets]
        else:
            target_names_to_use = target_names_for_eval


        for i in range(num_targets):
            target_name = target_names_to_use[i]
            y_true = y_true_multi[:, i]
            y_pred = y_pred_multi[:, i]

            y_true_flat = np.array(y_true).flatten()
            y_pred_flat = np.array(y_pred).flatten()

            valid_mask = ~np.isnan(y_pred_flat) & ~np.isinf(y_pred_flat) & \
                         ~np.isnan(y_true_flat) & ~np.isinf(y_true_flat)

            if not np.any(valid_mask): 
                print(f"警告: 指标 '{target_name}' 的所有预测值或真实值均为NaN/inf。")
                all_metrics_eval[target_name] = {
                    "MAPE": float('nan'), "RMSE": float('nan'),
                    "R2": float('nan'), "MAE": float('nan')
                }
                continue
                
            y_true_filtered = y_true_flat[valid_mask]
            y_pred_filtered = y_pred_flat[valid_mask]
            
            if len(y_true_filtered) == 0: 
                 print(f"警告: 指标 '{target_name}' 没有有效的预测值进行评估。")
                 all_metrics_eval[target_name] = {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')}
                 continue

            epsilon = 1e-8 
            mape_values = np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon))
            mape_values = np.where(np.isinf(mape_values), np.nan, mape_values) 
            mape = np.nanmean(mape_values) * 100 
            mape = min(mape, 1000.0) if pd.notna(mape) else float('nan')

            all_metrics_eval[target_name] = {
                "MAPE": mape,
                "RMSE": np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered)),
                "R2": r2_score(y_true_filtered, y_pred_filtered),
                "MAE": mean_absolute_error(y_true_filtered, y_pred_filtered)
            }
        return all_metrics_eval
    
    @staticmethod
    def static_target_names_for_eval():
        feature_names_static = [
            'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
            'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 
            'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 
            'CO', 'CO_24h'
        ]
        target_names_static = feature_names_static[:]
        if 'AQI' not in target_names_static:
            target_names_static.append('AQI')
        return target_names_static


    def plot_results(self, y_true_multi, y_pred_base_multi, y_pred_opt_multi):
        """绘制实际值与预测值的对比图 (多目标)。"""
        if y_true_multi is None:
            print("警告: 真实测试值 (y_true_multi) 为空，无法绘制结果图。")
            return
            
        num_targets = y_true_multi.shape[1]
        
        targets_to_plot_names = ['AQI', 'PM2.5', 'PM10'] 
        plot_indices = []
        actual_plot_target_names = []
        for name in targets_to_plot_names:
            try:
                idx = self.target_names.index(name)
                plot_indices.append(idx)
                actual_plot_target_names.append(name)
            except ValueError:
                print(f"警告: 目标 '{name}' 未在 self.target_names 中找到，无法绘制。")
        
        if not plot_indices:
            print("警告: 没有有效的目标可供绘制。")
            return

        num_plots = len(plot_indices)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots), sharex=True)
        if num_plots == 1:
            axes = [axes]

        fig.suptitle('部分指标预测结果对比', fontsize=16, y=1.02 if num_plots > 1 else 1.05)

        for k, target_idx in enumerate(plot_indices):
            ax = axes[k]
            current_target_name = actual_plot_target_names[k]

            y_true_flat = y_true_multi[:, target_idx].flatten()
            ax.plot(self.test_dates, y_true_flat, label=f'实际 {current_target_name}', color='blue', linewidth=1)
            
            if y_pred_base_multi is not None:
                y_pred_base_flat = y_pred_base_multi[:, target_idx].flatten()
                valid_base_pred_mask = ~np.isnan(y_pred_base_flat)
                if np.any(valid_base_pred_mask):
                    ax.plot(self.test_dates[valid_base_pred_mask], y_pred_base_flat[valid_base_pred_mask], label=f'基线预测 {current_target_name}', color='orange', alpha=0.7, linestyle='--')
            
            if y_pred_opt_multi is not None:
                y_pred_opt_flat = y_pred_opt_multi[:, target_idx].flatten()
                valid_opt_pred_mask = ~np.isnan(y_pred_opt_flat)
                if np.any(valid_opt_pred_mask):
                    ax.plot(self.test_dates[valid_opt_pred_mask], y_pred_opt_flat[valid_opt_pred_mask], label=f'优化预测 {current_target_name}', color='green', alpha=0.7, linestyle=':')
            
            ax.set_ylabel(current_target_name, fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.7)

        if pd.api.types.is_datetime64_any_dtype(self.test_dates):
             axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')) 
             plt.xticks(rotation=30, ha='right') 

        axes[-1].set_xlabel('时间', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97 if num_plots > 1 else 0.95])
        plt.show()


    def plot_metrics_comparison(self, eval_baseline_all_targets, eval_optimized_all_targets):
        """绘制基础模型和优化模型的评价指标对比图 (多目标)。"""
        metric_names_to_plot = ['MAPE', 'RMSE', 'R2', 'MAE']
        
        targets_for_metric_plot = ['AQI', 'PM2.5'] 
        
        num_metrics = len(metric_names_to_plot)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 3 * num_metrics), sharex=False)
        if num_metrics == 1:
            axes = [axes]

        fig.suptitle('关键指标的评估对比', fontsize=16, y=1.02)

        for i, metric_name in enumerate(metric_names_to_plot):
            ax = axes[i]
            baseline_values = []
            optimized_values = []
            plot_target_labels = []

            for target_label in targets_for_metric_plot:
                if target_label in eval_baseline_all_targets and target_label in eval_optimized_all_targets:
                    b_val = eval_baseline_all_targets[target_label].get(metric_name, float('nan'))
                    o_val = eval_optimized_all_targets[target_label].get(metric_name, float('nan'))
                    
                    baseline_values.append(b_val if pd.notna(b_val) else 0) 
                    optimized_values.append(o_val if pd.notna(o_val) else 0)
                    plot_target_labels.append(target_label)
            
            if not plot_target_labels: continue 

            x = np.arange(len(plot_target_labels))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', color='skyblue')
            rects2 = ax.bar(x + width/2, optimized_values, width, label='Optimized', color='lightcoral')

            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f'{metric_name} 对比', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(plot_target_labels, rotation=0, fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            def autolabel(rects, original_values_dict_for_metric):
                for k_rect, rect in enumerate(rects): # Renamed k to k_rect
                    height = rect.get_height()
                    target_for_label = plot_target_labels[k_rect] # Use k_rect
                    original_value = original_values_dict_for_metric.get(target_for_label, {}).get(metric_name, float('nan'))
                    display_text = f"{original_value:.2f}" if pd.notna(original_value) else "N/A"
                    
                    ax.annotate(display_text,
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7)
            
            autolabel(rects1, eval_baseline_all_targets)
            autolabel(rects2, eval_optimized_all_targets)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


def select_data_file_path(title="请选择数据集文件"):
    """打开文件对话框选择数据集路径。"""
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def select_model_file_path(title="请选择模型文件路径（用于加载或保存）", default_name="trained_model.keras"): 
    """打开文件对话框选择模型文件路径。"""
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.asksaveasfilename(
        title=title,
        initialfile=default_name,
        defaultextension=".keras", 
        filetypes=[("Keras 模型文件", "*.keras"), ("HDF5 模型文件", "*.h5"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def main(): 
    root = tk.Tk() 
    root.withdraw() 

    predict_future_choice = messagebox.askyesno("未来预测", "您想使用模型预测未来3天每小时的所有指标吗？")
    train_model_choice = messagebox.askyesno("模型训练", "您想训练或重新训练预测模型吗？")
    
    root.destroy() 

    data_path = select_data_file_path("请选择 AQI 数据集 Excel 文件")
    if not data_path:
        print("未选择数据集文件，程序退出。")
        return

    model_path_selected = select_model_file_path("请指定模型文件的加载/保存路径", "trained_multi_target_model.keras") 
    if not model_path_selected:
        print("未指定模型文件路径，程序退出。")
        return
    
    model_path = model_path_selected 

    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            print(f"已创建模型目录: {model_dir}")
        except OSError as e:
            print(f"错误: 创建模型目录 {model_dir} 失败: {e}。模型将保存在当前工作目录。")
            model_path = os.path.basename(model_path) 


    try:
        predictor = AQIPredictor(data_path, model_path)
    except Exception as e:
        print(f"初始化AQIPredictor失败: {e}")
        return 

    model_to_use = None
    y_test_inv = None 
    baseline_predictions = None 
    optimized_predictions = None 

    static_target_names = AQIPredictor.static_target_names_for_eval()
    eval_template = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}
    eval_baseline = {target: eval_template.copy() for target in static_target_names}
    eval_optimized = {target: eval_template.copy() for target in static_target_names}


    if predictor.y_test.size > 0 : 
        y_test_inv = predictor.scaler_y.inverse_transform(predictor.y_test)
        optimized_predictions = np.full_like(y_test_inv, np.nan) 


    if train_model_choice:
        print("\n--- 模型训练流程 ---")
        existing_model_for_retrain = predictor.load_existing_model()
        
        if existing_model_for_retrain:
            print("重新训练已加载的基线模型 (使用较小的学习率进行微调)...")
            optimizer = keras.optimizers.Adam(learning_rate=1e-4) 
            existing_model_for_retrain.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
            baseline_predictions, y_test_inv_train = predictor.train_and_evaluate(existing_model_for_retrain, epochs=10)
            model_to_use = existing_model_for_retrain
        else:
            print("构建并训练新的基线模型...")
            baseline_model_obj = predictor.build_model(activation_func='tanh', learning_rate=0.001)
            baseline_predictions, y_test_inv_train = predictor.train_and_evaluate(baseline_model_obj, epochs=50)
            model_to_use = baseline_model_obj 
        
        if y_test_inv_train is not None: y_test_inv = y_test_inv_train 
        predictor.save_model(model_to_use)
        if y_test_inv is not None and baseline_predictions is not None:
            eval_baseline = AQIPredictor.evaluate(y_test_inv, baseline_predictions)
        
        # 如果不运行优化，optimized_predictions 保持为NaN，eval_optimized 保持为NaN字典
        # 只有在优化成功时才更新它们
        if y_test_inv is not None: # 确保 y_test_inv 有效
            optimized_predictions = np.full_like(y_test_inv, np.nan) # 重新初始化为NaN
        eval_optimized = {target: eval_template.copy() for target in static_target_names}


        root = tk.Tk()
        root.withdraw()
        run_optimization = messagebox.askyesno("模型优化", "基线模型已训练。\n您想运行贝叶斯优化以尝试改进模型吗？\n（这可能需要一些时间）")
        root.destroy()

        if run_optimization:
            print("\n--- 优化模型 (使用贝叶斯优化) ---")
            optimized_model_tuned = predictor.tune_model() 
            if optimized_model_tuned:
                print("训练和评估优化后的模型...")
                if not optimized_model_tuned.optimizer:
                    print("编译优化后的模型...")
                    lr_opt = 1e-3 
                    if hasattr(optimized_model_tuned, 'optimizer') and hasattr(optimized_model_tuned.optimizer, 'learning_rate'):
                        try:
                            lr_opt = optimized_model_tuned.optimizer.learning_rate.numpy()
                        except: pass 
                    opt_optimizer = keras.optimizers.Adam(learning_rate=lr_opt)
                    optimized_model_tuned.compile(optimizer=opt_optimizer, loss='mean_squared_error', metrics=['mae'])
                
                # 获取优化模型的预测
                current_optimized_predictions, _ = predictor.train_and_evaluate(optimized_model_tuned, epochs=50)
                if y_test_inv is not None and current_optimized_predictions is not None:
                    eval_optimized = AQIPredictor.evaluate(y_test_inv, current_optimized_predictions)
                    optimized_predictions = current_optimized_predictions # 只有成功时才更新
                model_to_use = optimized_model_tuned 
            else:
                print("优化失败。后续操作将使用之前训练的基线模型。")
                # optimized_predictions 和 eval_optimized 保持为初始的NaN状态
    
    else: 
        print("\n--- 加载现有模型 ---")
        model_to_use = predictor.load_existing_model()
        if model_to_use:
            print("使用已加载的模型进行评估...")
            if not model_to_use.optimizer:
                print("编译已加载的模型...")
                model_to_use.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
            
            if predictor.X_test.size > 0:
                scaled_preds = model_to_use.predict(predictor.X_test)
                baseline_predictions = predictor.scaler_y.inverse_transform(scaled_preds)
                if y_test_inv is not None and baseline_predictions is not None:
                     eval_baseline = AQIPredictor.evaluate(y_test_inv, baseline_predictions)
            else:
                print("警告: 测试集为空，无法进行评估。")
            # optimized_predictions 和 eval_optimized 保持为初始的NaN状态
        else:
            messagebox.showerror("错误", f"在 {model_path} 未找到模型，且未选择训练新模型。无法继续。")
            print(f"在 {model_path} 未找到模型，且未选择训练新模型。程序退出。")
            return

    if y_test_inv is None: 
        if predictor.y_test.size > 0:
            y_test_inv = predictor.scaler_y.inverse_transform(predictor.y_test)
        else:
            print("错误: 测试集真实值 (y_test_inv) 不可用，无法进行评估和绘图。")
            return


    print("\n--- 评估结果 ---")
    print("\n基线模型评估:")
    for target_name, metrics in eval_baseline.items():
        print(f"  指标 {target_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
    
    print("\n优化模型评估:") 
    for target_name, metrics in eval_optimized.items():
        print(f"  指标 {target_name}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}") 

    print("\n--- 生成图表 ---")
    if baseline_predictions is None and y_test_inv is not None: 
        baseline_predictions = np.full_like(y_test_inv, np.nan) 

    if y_test_inv is not None: 
        predictor.plot_results(y_test_inv, baseline_predictions, optimized_predictions)
        predictor.plot_metrics_comparison(eval_baseline, eval_optimized)
    else:
        print("警告: 由于缺少测试集真实值，无法生成图表。")


    if predict_future_choice:
        if model_to_use:
            print("\n--- 未来AQI预测 ---")
            predictor.predict_future(model_to_use, num_steps=72)
        else:
            messagebox.showwarning("未来预测", "没有可用的模型进行未来预测（未训练或加载失败）。")
            print("没有可用的模型进行未来预测。")

if __name__ == "__main__":
    main()
