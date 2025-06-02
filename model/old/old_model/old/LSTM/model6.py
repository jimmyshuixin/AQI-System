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

        参数:
        - data_path: 数据集路径 (应为 .xlsx 文件)
        - model_path: 模型保存或加载路径（默认值为'trained_model.h5'）
        """
        self.data_path = data_path  # 保存数据集路径
        self.model_path = model_path  # 保存模型文件路径
        self.scaler_X, self.scaler_y = MinMaxScaler(), MinMaxScaler()  # 初始化输入和输出的归一化工具
        # 更新特征列名列表
        self.feature_names = [
            'PM2.5', 'PM2.5_24h', 'PM10', 'PM10_24h', 
            'SO2', 'SO2_24h', 'NO2', 'NO2_24h', 
            'O3', 'O3_24h', 'O3_8h', 'O3_8h_24h', 
            'CO', 'CO_24h'
        ]
        self.target_name = 'AQI' # 定义目标列名
        self.date_col_name = 'date' # 定义日期列名
        self.hour_col_name = 'hour' # 定义小时列名
        self.true_absolute_last_datetime = None # 存储原始数据文件中的真实最后时间戳
        self.last_datetime_with_complete_features = None # 存储清理后数据中最后一个具有完整特征的时间戳
        self.X_scaled_all = None # 用于存储所有缩放后的特征数据，以便未来预测使用最后一个已知特征
        self._load_and_prepare_data()  # 加载并预处理数据
    
    def _load_and_prepare_data(self):
        """加载并预处理数据。"""
        try:
            # 读取由 aqi_data_extractor_script 生成的 Excel 文件
            aqi_data_original = pd.read_excel(self.data_path)
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.data_path} 未找到。请检查路径。")
            raise
        except Exception as e:
            print(f"错误: 读取数据文件 {self.data_path} 时发生错误: {e}")
            raise

        if aqi_data_original.empty:
            raise ValueError("错误: 读取的原始数据文件为空。")

        # 检查所需列是否存在于原始数据中
        original_cols = aqi_data_original.columns
        missing_features_orig = [col for col in self.feature_names if col not in original_cols]
        if missing_features_orig:
            print(f"错误: 原始数据文件中缺少以下特征列: {', '.join(missing_features_orig)}")
            raise ValueError("原始数据文件中缺少必要的特征列。")
        if self.target_name not in original_cols:
            print(f"错误: 原始数据文件中缺少目标列: {self.target_name}")
            raise ValueError("原始数据文件中缺少目标列。")
        if self.date_col_name not in original_cols:
            raise ValueError(f"原始数据文件中缺少日期列: {self.date_col_name}。")
        if self.hour_col_name not in original_cols:
            raise ValueError(f"原始数据文件中缺少小时列: {self.hour_col_name}。")

        # 在进行任何修改或dropna之前，先处理datetime并获取原始的最后时间戳
        try:
            aqi_data_original['datetime_str_orig'] = aqi_data_original[self.date_col_name].astype(str) + \
                                                     aqi_data_original[self.hour_col_name].astype(str).str.zfill(2)
            aqi_data_original['datetime'] = pd.to_datetime(aqi_data_original['datetime_str_orig'], format='%Y%m%d%H')
            # 按datetime排序，以确保iloc[-1]确实是最后一个时间点
            aqi_data_original.sort_values(by='datetime', inplace=True)
            self.true_absolute_last_datetime = aqi_data_original['datetime'].iloc[-1]
            print(f"信息: 原始数据文件中的真实最后时间戳为: {self.true_absolute_last_datetime}")
        except Exception as e:
            print(f"错误: 在原始数据中转换日期和小时列为datetime对象时失败: {e}")
            raise

        # 现在创建工作副本进行清理
        aqi_data = aqi_data_original.copy()

        # --- NaN 值处理：使用前向填充后接后向填充进行插补 ---
        cols_to_impute = self.feature_names + [self.target_name]
        nan_counts_before = aqi_data[cols_to_impute].isnull().sum()
        
        if nan_counts_before.sum() > 0:
            print(f"信息: 在插补前，特征和目标列中存在以下数量的NaN值:\n{nan_counts_before[nan_counts_before > 0]}")
            # 对每个需要插补的列分别进行ffill().bfill()，以避免数据类型问题
            for col in cols_to_impute:
                if aqi_data[col].isnull().any():
                    aqi_data[col] = aqi_data[col].ffill().bfill()
            
            nan_counts_after = aqi_data[cols_to_impute].isnull().sum()
            if nan_counts_after.sum() > 0:
                print(f"警告: 插补后仍然存在NaN值，请检查数据源或插补策略:\n{nan_counts_after[nan_counts_after > 0]}")
                # 如果插补后仍有NaN，则移除这些行作为最后手段
                aqi_data.dropna(subset=cols_to_impute, inplace=True)
                print("信息: 已移除插补后仍包含NaN的行。")
            else:
                print("信息: 所有NaN值已通过ffill().bfill()成功插补。")
        else:
            print("信息: 特征和目标列中没有检测到NaN值。")
        # --- 结束 NaN 值处理 ---
        
        if aqi_data.empty:
            raise ValueError("数据在插补（或后续移除）NaN值后为空。请检查输入数据。")
        
        # 保存具有完整特征的最后一个历史时间戳 (现在应该与 true_absolute_last_datetime 相同，如果所有行都保留了)
        self.last_datetime_with_complete_features = aqi_data['datetime'].iloc[-1]
        print(f"信息: 插补/清理后数据中，用于模型特征的最后一个时间戳为: {self.last_datetime_with_complete_features}")


        X, y = aqi_data[self.feature_names], aqi_data[self.target_name].values.reshape(-1, 1)

        # 数据标准化
        self.X_scaled_all = self.scaler_X.fit_transform(X) # 保存所有缩放后的X，用于未来预测
        self.y_scaled = self.scaler_y.fit_transform(y)

        if np.isnan(self.X_scaled_all).any():
            print("警告: 特征数据 X_scaled_all 在归一化后包含NaN值。")
        if np.isnan(self.y_scaled).any():
            print("警告: 目标数据 y_scaled 在归一化后包含NaN值。")

        
        # 数据划分
        train_size = int(len(self.X_scaled_all) * 0.8)
        if train_size == 0 or (len(self.X_scaled_all) - train_size) == 0 : # 确保训练集和测试集都有数据
             raise ValueError(f"数据量太小 (总共 {len(self.X_scaled_all)} 行，训练集 {train_size} 行)，无法进行有效的训练集和测试集划分。")

        self.X_train, self.X_test = self.X_scaled_all[:train_size], self.X_scaled_all[train_size:]
        self.y_train, self.y_test = self.y_scaled[:train_size], self.y_scaled[train_size:]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

        # 保存测试集的日期，用于绘图
        self.test_dates = aqi_data['datetime'].iloc[train_size:].reset_index(drop=True)


    def build_model(self, units=50, dropout_rate=0.2, activation_func='tanh', learning_rate=0.001):
        """
        创建并编译双向LSTM模型。

        参数:
        - units: LSTM层神经元数量
        - dropout_rate: Dropout层丢弃率
        - activation_func: LSTM层的激活函数 (默认为 'tanh')
        - learning_rate: 优化器的学习率

        返回:
        - model: 编译后的Keras模型
        """
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        x = Bidirectional(LSTM(units=units, activation=activation_func))(inputs) # 使用传入的激活函数
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        return model

    def train_and_evaluate(self, model, epochs=50):
        """
        训练模型并评估在测试集上的表现。
        假设模型在传入此方法前已经编译好。
        """
        print(f"开始在 {epochs} 个周期内训练模型...")
        model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
        print("模型训练完成。正在进行预测...")
        predictions_scaled = model.predict(self.X_test)

        if np.isnan(predictions_scaled).any():
            nan_count = np.sum(np.isnan(predictions_scaled))
            print(f"警告: 模型预测结果 (缩放后) 包含 {nan_count} 个NaN值。")
            if np.isnan(predictions_scaled).all():
                print("警告: 模型预测结果 (缩放后) 全部为 NaN。")
                return np.full_like(self.y_test, np.nan), self.scaler_y.inverse_transform(self.y_test)
        
        predictions_inversed = self.scaler_y.inverse_transform(predictions_scaled)
        y_test_inversed = self.scaler_y.inverse_transform(self.y_test)

        if np.isnan(predictions_inversed).any():
            nan_count_inv = np.sum(np.isnan(predictions_inversed))
            print(f"警告: 模型预测结果 (逆缩放后) 包含 {nan_count_inv} 个NaN值。")
            if np.isnan(predictions_inversed).all():
                print("警告: 模型预测结果 (逆缩放后) 全部为 NaN。")
        
        if np.isnan(y_test_inversed).any():
            print(f"警告: 测试集真实值 (逆缩放后) 包含NaN值。数量: {np.sum(np.isnan(y_test_inversed))}")


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
        """加载已保存的模型。"""
        load_path = self.model_path
        if not os.path.exists(load_path) and not load_path.endswith(".keras"):
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                print(f"信息: 未找到 {load_path}，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                load_path = keras_format_path
            else:
                print(f"模型文件 {self.model_path} (或 .keras 变体) 未找到。")
                return None
        elif not os.path.exists(load_path):
             print(f"模型文件 {load_path} 未找到。")
             return None


        print(f"加载已保存的模型：{load_path}")
        try:
            return load_model(load_path) 
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
        使用最后一个已知的特征集预测未来指定步数的AQI。
        """
        if self.X_scaled_all is None or self.X_scaled_all.shape[0] == 0:
            print("错误: 没有可用于未来预测的已缩放数据 (X_scaled_all)。")
            return None
        
        # 获取最后一个已知的特征序列 (在 reshape 成 LSTM 输入之前)
        # 这个特征集对应 self.last_datetime_with_complete_features
        last_known_features_scaled = self.X_scaled_all[-1, :] 
        # Reshape for LSTM: (1, 1, num_features)
        current_input = last_known_features_scaled.reshape(1, 1, self.X_scaled_all.shape[1])
        
        future_predictions_scaled_list = []

        print(f"\n开始预测未来 {num_steps} 小时的 AQI...")
        
        # 使用 self.true_absolute_last_datetime 来确定预测的起始时间
        if self.true_absolute_last_datetime is None:
            print("错误: 未能确定原始数据文件的真实最后一个历史时间戳，无法准确生成未来预测的时间序列。")
            if self.last_datetime_with_complete_features is not None:
                print(f"警告: 将使用具有完整特征的最后一个时间戳 {self.last_datetime_with_complete_features} 作为未来预测的基准。")
                start_prediction_datetime = self.last_datetime_with_complete_features
            else: 
                print("错误: 无法确定任何有效的基准时间戳进行未来预测。")
                return None
        else:
            start_prediction_datetime = self.true_absolute_last_datetime


        future_timestamps = pd.date_range(
            start=start_prediction_datetime + pd.Timedelta(hours=1), 
            periods=num_steps, 
            freq='h' # 使用小写 'h'
        )
        print(f"信息: 未来预测将从 {future_timestamps[0]} (基于 {start_prediction_datetime} 之后一小时) 开始，直到 {future_timestamps[-1]}。")
        print(f"信息: 用于预测的最后一个已知特征集对应的时间戳是 {self.last_datetime_with_complete_features}。")

        # 注意：对于非自回归模型，使用相同的 last_known_features_scaled 进行所有未来步的预测
        # 将导致所有未来预测值相同。这是一个已知的限制，除非模型设计为自回归
        # 或有未来的外生变量可用。
        print("信息: 当前模型为非自回归模型，未来预测将基于最后一个已知特征集。")

        for i in range(num_steps):
            prediction_step_scaled = model.predict(current_input, verbose=0)
            future_predictions_scaled_list.append(prediction_step_scaled[0, 0])
        
        future_predictions_scaled_np = np.array(future_predictions_scaled_list).reshape(-1, 1)
        future_predictions_inversed = self.scaler_y.inverse_transform(future_predictions_scaled_np)
        
        df_future = pd.DataFrame({'Timestamp': future_timestamps, 'Predicted_AQI': future_predictions_inversed.flatten()})
        
        print("\n未来3天每小时AQI预测:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
            print(df_future)

        plt.figure(figsize=(12, 6))
        plt.plot(df_future['Timestamp'], df_future['Predicted_AQI'], label='未来预测 AQI', color='red', marker='.', linestyle='-')
        plt.title('未来3天每小时 AQI 预测', fontsize=16)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('预测 AQI', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.show()

        return df_future


    @staticmethod
    def evaluate(y_true, y_pred):
        """计算评价指标。"""
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()

        valid_mask = ~np.isnan(y_pred_flat) & ~np.isinf(y_pred_flat) & \
                     ~np.isnan(y_true_flat) & ~np.isinf(y_true_flat)

        if not np.any(valid_mask): 
            print("警告: 所有预测值或真实值均为NaN/inf，无法计算评估指标。")
            return {
                "MAPE": float('nan'), "RMSE": float('nan'),
                "R2": float('nan'), "MAE": float('nan')
            }
            
        y_true_filtered = y_true_flat[valid_mask]
        y_pred_filtered = y_pred_flat[valid_mask]
        
        if len(y_true_filtered) == 0: 
             print("警告: 没有有效的预测值进行评估（在过滤NaN/inf后）。")
             return {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')}

        epsilon = 1e-8 
        mape_values = np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon))
        mape_values = np.where(np.isinf(mape_values), np.nan, mape_values) 
        mape = np.nanmean(mape_values) * 100 
        mape = min(mape, 1000.0) if pd.notna(mape) else float('nan')


        return {
            "MAPE": mape,
            "RMSE": np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered)),
            "R2": r2_score(y_true_filtered, y_pred_filtered),
            "MAE": mean_absolute_error(y_true_filtered, y_pred_filtered)
        }

    def plot_results(self, y_true, y_pred_base, y_pred_opt):
        """绘制实际值与预测值的对比图。"""
        plt.figure(figsize=(14, 7))
        
        y_true_flat = y_true.flatten() 

        plt.plot(self.test_dates, y_true_flat, label='实际 AQI', color='blue', linewidth=1)
        
        if y_pred_base is not None:
            y_pred_base_flat = y_pred_base.flatten()
            valid_base_pred_mask = ~np.isnan(y_pred_base_flat)
            if np.any(valid_base_pred_mask):
                plt.plot(self.test_dates[valid_base_pred_mask], y_pred_base_flat[valid_base_pred_mask], label='基线模型预测 AQI', color='orange', alpha=0.7, linestyle='--')
        
        if y_pred_opt is not None:
            y_pred_opt_flat = y_pred_opt.flatten()
            valid_opt_pred_mask = ~np.isnan(y_pred_opt_flat)
            if np.any(valid_opt_pred_mask):
                plt.plot(self.test_dates[valid_opt_pred_mask], y_pred_opt_flat[valid_opt_pred_mask], label='优化模型预测 AQI', color='green', alpha=0.7, linestyle=':')
        
        if pd.api.types.is_datetime64_any_dtype(self.test_dates):
             plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')) 
             plt.xticks(rotation=30, ha='right') 

        plt.title('AQI 预测结果对比', fontsize=16)
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout() 
        plt.show()

    def plot_metrics_comparison(self, eval_baseline, eval_optimized):
        """绘制基础模型和优化模型的评价指标对比图。"""
        metrics_data = {
            'Baseline': eval_baseline,
            'Optimized': eval_optimized
        }
        metric_names = ['MAPE', 'RMSE', 'R2', 'MAE']
        
        plot_data = {}
        for model_type, metrics in metrics_data.items():
            plot_data[model_type] = {
                k: (metrics[k] if pd.notna(metrics[k]) else (1000 if k == 'MAPE' else 0) ) 
                for k in metric_names
            }


        df_metrics = pd.DataFrame(plot_data).T[metric_names] 

        ax = df_metrics.plot(kind='bar', figsize=(12, 8), colormap='viridis', width=0.8) 
        
        plt.title('模型评价指标对比 (MAPE, RMSE, MAE越低越好, R2越高越好)', fontsize=16) 
        plt.ylabel('指标值', fontsize=12)
        plt.xticks(rotation=0, fontsize=10) 
        plt.legend(title='模型', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 调整文本标签的定位逻辑，以确保它们在柱子内部或上方，并且清晰可见
        for i, model_name in enumerate(df_metrics.index): 
            for j, metric_name in enumerate(df_metrics.columns): 
                original_value = metrics_data[model_name][metric_name] 
                display_value_text = f"{original_value:.3f}" if pd.notna(original_value) else "N/A"
                
                bar_container_index = j 
                if len(ax.containers) <= bar_container_index: 
                    continue
                bar_container = ax.containers[bar_container_index]
                
                if len(bar_container.patches) <= i: 
                    continue
                bar = bar_container.patches[i]
                
                bar_height = bar.get_height()
                bar_x = bar.get_x() + bar.get_width() / 2.0

                # 根据柱子高度调整文本Y位置
                if pd.notna(bar_height) and bar_height > 0 :
                    text_y_position = bar_height * 0.95 
                    va_align = 'top'
                    if bar_height < (0.1 * df_metrics[metric_name].abs().max()): 
                        text_y_position = bar_height + 0.01 * df_metrics[metric_name].abs().max() 
                        va_align = 'bottom'
                elif pd.notna(bar_height) and bar_height < 0: 
                    text_y_position = bar_height * 1.05
                    va_align = 'bottom'
                else: 
                    text_y_position = 0.01
                    va_align = 'bottom'


                plt.text(
                    bar_x, 
                    text_y_position,
                    display_value_text, 
                    ha='center',
                    va=va_align,
                    fontsize=8, 
                    color='black',
                    bbox=dict(facecolor='white', alpha=0.5, pad=1, boxstyle='round,pad=0.2') 
                )
        
        plt.tight_layout()
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

    predict_future_choice = messagebox.askyesno("未来预测", "您想使用模型预测未来3天每小时的AQI吗？")
    train_model_choice = messagebox.askyesno("模型训练", "您想训练或重新训练AQI预测模型吗？")
    
    root.destroy() 

    data_path = select_data_file_path("请选择 AQI 数据集 Excel 文件")
    if not data_path:
        print("未选择数据集文件，程序退出。")
        return

    model_path_selected = select_model_file_path("请指定模型文件的加载/保存路径", "trained_model.keras") 
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
    # 初始化 optimized_predictions 为与 y_test_inv 形状相同的NaN数组，如果 y_test_inv 可用
    if predictor.y_test.size > 0:
        y_test_inv = predictor.scaler_y.inverse_transform(predictor.y_test)
        optimized_predictions = np.full_like(y_test_inv, np.nan)
    else:
        optimized_predictions = None # 如果没有测试数据，则为None

    eval_baseline = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}
    eval_optimized = {metric: float('nan') for metric in ['MAPE', 'RMSE', 'R2', 'MAE']}


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
            baseline_model = predictor.build_model(activation_func='tanh', learning_rate=0.001)
            baseline_predictions, y_test_inv_train = predictor.train_and_evaluate(baseline_model, epochs=50)
            model_to_use = baseline_model
        
        if y_test_inv_train is not None: y_test_inv = y_test_inv_train 
        predictor.save_model(model_to_use)
        if y_test_inv is not None and baseline_predictions is not None:
            eval_baseline = AQIPredictor.evaluate(y_test_inv, baseline_predictions)
        
        # 如果没有选择优化，optimized_predictions 保持为NaN数组，eval_optimized 保持为NaN字典
        # 这样绘图时，优化模型的线和指标将不会显示或显示为N/A

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
                
                # 重新获取 optimized_predictions
                optimized_predictions, _ = predictor.train_and_evaluate(optimized_model_tuned, epochs=50)
                if y_test_inv is not None and optimized_predictions is not None:
                    eval_optimized = AQIPredictor.evaluate(y_test_inv, optimized_predictions)
                model_to_use = optimized_model_tuned 
            else:
                print("优化失败。后续操作将使用之前训练的基线模型。")
                # optimized_predictions 保持为NaN数组, eval_optimized 保持为NaN字典
    
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
            # optimized_predictions 保持为NaN数组, eval_optimized 保持为NaN字典
        else:
            messagebox.showerror("错误", f"在 {model_path} 未找到模型，且未选择训练新模型。无法继续。")
            print(f"在 {model_path} 未找到模型，且未选择训练新模型。程序退出。")
            return

    if y_test_inv is None: # 再次检查，以防万一
        if predictor.y_test.size > 0:
            y_test_inv = predictor.scaler_y.inverse_transform(predictor.y_test)
        else:
            print("错误: 测试集真实值 (y_test_inv) 不可用，无法进行评估和绘图。")
            return


    print("\n--- 评估结果 ---")
    print("\n基线模型评估:")
    for metric, value in eval_baseline.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n优化模型评估:") 
    for metric, value in eval_optimized.items():
        print(f"{metric}: {value:.4f}") # 如果未优化，将打印NaN

    print("\n--- 生成图表 ---")
    # 确保 baseline_predictions 至少被初始化 (如果加载模型但测试集为空，它可能是None)
    if baseline_predictions is None and y_test_inv is not None:
        baseline_predictions = np.full_like(y_test_inv, np.nan)

    if y_test_inv is not None: # 检查 y_test_inv 是否有效
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
