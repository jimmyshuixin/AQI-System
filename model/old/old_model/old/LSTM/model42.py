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
from tkinter import filedialog # 从tkinter导入filedialog模块
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
        self._load_and_prepare_data()  # 加载并预处理数据
    
    def _load_and_prepare_data(self):
        """加载并预处理数据。"""
        try:
            # 读取由 aqi_data_extractor_script 生成的 Excel 文件
            aqi_data = pd.read_excel(self.data_path)
            if aqi_data[self.feature_names + [self.target_name]].isnull().values.any():
                print(f"警告: 原始数据文件 {self.data_path} 在读取后包含NaN值。将进行移除。")
        except FileNotFoundError:
            print(f"错误: 数据文件 {self.data_path} 未找到。请检查路径。")
            raise
        except Exception as e:
            print(f"错误: 读取数据文件 {self.data_path} 时发生错误: {e}")
            raise

        # 检查所需列是否存在
        missing_features = [col for col in self.feature_names if col not in aqi_data.columns]
        if missing_features:
            print(f"错误: 数据文件中缺少以下特征列: {', '.join(missing_features)}")
            raise ValueError("数据文件中缺少必要的特征列。")
        
        if self.target_name not in aqi_data.columns:
            print(f"错误: 数据文件中缺少目标列: {self.target_name}")
            raise ValueError("数据文件中缺少目标列。")

        # 检查日期和小时列是否存在
        if self.date_col_name not in aqi_data.columns:
            raise ValueError(f"数据文件中缺少日期列: {self.date_col_name}。")
        if self.hour_col_name not in aqi_data.columns:
            raise ValueError(f"数据文件中缺少小时列: {self.hour_col_name}。")

        # 合并 'date' 和 'hour' 列来创建 datetime 对象
        try:
            # 确保 'date' 是字符串格式YYYYMMDD，'hour' 是可以补零的数字或字符串
            aqi_data['datetime_str'] = aqi_data[self.date_col_name].astype(str) + \
                                       aqi_data[self.hour_col_name].astype(str).str.zfill(2)
            aqi_data['datetime'] = pd.to_datetime(aqi_data['datetime_str'], format='%Y%m%d%H')
        except Exception as e:
            print(f"错误: 转换日期和小时列为datetime对象时失败: {e}")
            print("请确保 'date' 列是YYYYMMDD 格式，'hour' 列是 0-23 范围内的小时数。")
            raise
        
        # 移除包含NaN的行 (在特征或目标中)
        initial_rows = len(aqi_data)
        aqi_data.dropna(subset=self.feature_names + [self.target_name], inplace=True)
        if initial_rows > len(aqi_data):
            print(f"信息: 从数据中移除了 {initial_rows - len(aqi_data)} 行，因为它们在特征或目标列中包含NaN值。")
        if aqi_data.empty:
            raise ValueError("数据在移除NaN值后为空。请检查输入数据。")


        X, y = aqi_data[self.feature_names], aqi_data[self.target_name].values.reshape(-1, 1)

        # 数据标准化
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)

        if np.isnan(self.X_scaled).any():
            print("警告: 特征数据 X_scaled 在归一化后包含NaN值。")
        if np.isnan(self.y_scaled).any():
            print("警告: 目标数据 y_scaled 在归一化后包含NaN值。")

        
        # 数据划分
        train_size = int(len(self.X_scaled) * 0.8)
        if train_size == 0 or (len(self.X_scaled) - train_size) == 0 : # 确保训练集和测试集都有数据
             raise ValueError(f"数据量太小 (总共 {len(self.X_scaled)} 行，训练集 {train_size} 行)，无法进行有效的训练集和测试集划分。")

        self.X_train, self.X_test = self.X_scaled[:train_size], self.X_scaled[train_size:]
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
            # 推荐使用 .keras 格式保存模型
            save_path = self.model_path
            if not save_path.endswith(".keras"):
                save_path = os.path.splitext(save_path)[0] + ".keras"
                print(f"信息: 模型将以 .keras 格式保存到: {save_path}")
            
            model.save(save_path) # 使用 .keras 格式
            print(f"模型已保存为 {save_path}")
            # 如果仍然需要 .h5 用于旧版兼容性，可以额外保存，但首选 .keras
            # model.save(self.model_path) # 保存为 .h5 (如果 self.model_path 以 .h5 结尾)
            # print(f"模型也已保存为 (HDF5): {self.model_path}")


        except Exception as e:
            print(f"错误: 保存模型失败: {e}")


    def load_existing_model(self):
        """加载已保存的模型。"""
        load_path = self.model_path
        if not os.path.exists(load_path) and not load_path.endswith(".keras"):
            # 尝试 .keras 格式如果 .h5 不存在
            keras_format_path = os.path.splitext(load_path)[0] + ".keras"
            if os.path.exists(keras_format_path):
                print(f"信息: 未找到 {load_path}，但找到了 {keras_format_path}。将加载 .keras 格式模型。")
                load_path = keras_format_path
            else:
                print(f"模型文件 {self.model_path} (或 .keras 变体) 未找到，需重新训练模型。")
                return None
        elif not os.path.exists(load_path):
             print(f"模型文件 {load_path} 未找到，需重新训练模型。")
             return None


        print(f"加载已保存的模型：{load_path}")
        try:
            return load_model(load_path) 
        except Exception as e:
            print(f"错误: 加载模型 {load_path} 失败: {e}。将尝试训练新模型。")
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
                activation_func = 'tanh' # 固定为 tanh 以提高稳定性

                # build_model 现在接受 learning_rate
                model = self.build_model(units, dropout_rate, activation_func=activation_func, learning_rate=learning_rate) 
                # model.compile 已经在 build_model 中完成
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
            tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, verbose=1) # verbose=1 for more tuner output
            
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
            # 确保优化模型已编译 (Keras Tuner 通常返回已编译模型)
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


    @staticmethod
    def evaluate(y_true, y_pred):
        """计算评价指标。"""
        y_true_flat = np.array(y_true).flatten()
        y_pred_flat = np.array(y_pred).flatten()

        # 同时检查 y_true 和 y_pred 中的 NaN 和 inf
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

        # 避免除以零的 MAPE
        epsilon = 1e-8 # 一个非常小的数，以避免除以零
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon))) * 100
        # Clamp MAPE to a reasonable upper bound if necessary, e.g., 1000, if extreme outliers cause issues
        mape = min(mape, 1000.0)


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
        
        y_pred_base_flat = y_pred_base.flatten()
        valid_base_pred_mask = ~np.isnan(y_pred_base_flat)
        if np.any(valid_base_pred_mask):
            plt.plot(self.test_dates[valid_base_pred_mask], y_pred_base_flat[valid_base_pred_mask], label='基线模型预测 AQI', color='orange', alpha=0.7, linestyle='--')
        
        y_pred_opt_flat = y_pred_opt.flatten()
        valid_opt_pred_mask = ~np.isnan(y_pred_opt_flat)
        if np.any(valid_opt_pred_mask):
            plt.plot(self.test_dates[valid_opt_pred_mask], y_pred_opt_flat[valid_opt_pred_mask], label='优化模型预测 AQI', color='green', alpha=0.7, linestyle=':')
        
        if pd.api.types.is_datetime64_any_dtype(self.test_dates):
             plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')) 
             plt.xticks(rotation=30, ha='right') # 调整旋转角度和对齐

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
            # 用一个较大的值（例如，如果MAPE非常大）或0来替换NaN，以便绘图
            # 但标签将显示真实值或N/A
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
        
        for i, model_name in enumerate(df_metrics.index): 
            for j, metric_name in enumerate(df_metrics.columns): 
                original_value = metrics_data[model_name][metric_name] 
                display_value_text = f"{original_value:.3f}" if pd.notna(original_value) else "N/A"
                
                bar_container = ax.containers[j] # 获取对应指标的柱子容器
                bar = bar_container.patches[i] # 获取当前模型的柱子
                
                bar_height = bar.get_height()
                bar_x = bar.get_x() + bar.get_width() / 2.0

                text_y_position = bar_height * 1.02 if pd.notna(bar_height) and bar_height >=0 else 0.01 # 稍微在柱子上方
                if pd.isna(bar_height) or bar_height < 0:
                    text_y_position = 0.01


                plt.text(
                    bar_x, 
                    text_y_position,
                    display_value_text, 
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black'
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

def select_model_file_path(title="请选择模型文件路径（用于加载或保存）", default_name="trained_model.keras"): # 默认 .keras
    """打开文件对话框选择模型文件路径。"""
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.asksaveasfilename(
        title=title,
        initialfile=default_name,
        defaultextension=".keras", # 默认 .keras
        filetypes=[("Keras 模型文件", "*.keras"), ("HDF5 模型文件", "*.h5"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def main(): 
    data_path = select_data_file_path("请选择 AQI 数据集 Excel 文件")
    if not data_path:
        print("未选择数据集文件，程序退出。")
        return

    model_path_selected = select_model_file_path("请指定模型文件的加载/保存路径", "trained_model.keras") # 默认 .keras
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

    print("\n--- 基线模型 ---")
    baseline_model = predictor.load_existing_model()
    if baseline_model is None:
        print("构建新的基线模型...")
        baseline_model = predictor.build_model(activation_func='tanh', learning_rate=0.001) # 明确学习率
    else:
        print("重新编译已加载的基线模型以用于训练/评估 (使用较小的学习率)...")
        # 使用较小的学习率进行微调
        optimizer = keras.optimizers.Adam(learning_rate=1e-4) 
        baseline_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    print("训练和评估基线模型...")
    # 对已加载并重新编译的模型，可能只需要少量周期的微调
    baseline_epochs = 10 if predictor.load_existing_model() and baseline_model is not None else 50
    baseline_predictions, y_test_inv = predictor.train_and_evaluate(baseline_model, epochs=baseline_epochs)
    predictor.save_model(baseline_model)

    print("\n--- 优化模型 (使用贝叶斯优化) ---")
    optimized_model = predictor.tune_model() 
    
    if optimized_model is None: 
        print("警告: 模型优化失败或未执行，将使用默认参数的 tanh 模型进行评估。")
        optimized_model = predictor.build_model(activation_func='tanh', learning_rate=0.001)
        if not optimized_model.optimizer: # 确保编译
             optimized_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])


    print("训练和评估优化后的模型...")
    if not optimized_model.optimizer: # 再次确保编译
        print("编译优化（或回退的默认）模型...")
        # 如果是从 tuner 获取的，它应该已经编译了。这是针对 build_model 回退的情况。
        lr = optimized_model.optimizer.learning_rate.numpy() if hasattr(optimized_model.optimizer, 'learning_rate') else 0.001
        opt = keras.optimizers.Adam(learning_rate=lr)
        optimized_model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])
    optimized_predictions, _ = predictor.train_and_evaluate(optimized_model, epochs=50) # 优化模型使用标准epochs


    print("\n--- 评估结果 ---")
    eval_baseline = AQIPredictor.evaluate(y_test_inv, baseline_predictions)
    eval_optimized = AQIPredictor.evaluate(y_test_inv, optimized_predictions) 


    print("\n基线模型评估:")
    for metric, value in eval_baseline.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n优化模型评估:")
    for metric, value in eval_optimized.items():
        print(f"{metric}: {value:.4f}")

    print("\n--- 生成图表 ---")
    predictor.plot_results(y_test_inv, baseline_predictions, optimized_predictions)
    predictor.plot_metrics_comparison(eval_baseline, eval_optimized)

if __name__ == "__main__":
    main()
