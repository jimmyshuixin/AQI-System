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
# import tensorflow as tf # Optional: for tf.config.run_functions_eagerly(True) if needed for other issues
# tf.config.run_functions_eagerly(True) # Uncomment for debugging certain TensorFlow/Keras eager execution issues

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
        aqi_data.dropna(subset=self.feature_names + [self.target_name], inplace=True)
        if aqi_data.empty:
            raise ValueError("数据在移除NaN值后为空。请检查输入数据。")


        X, y = aqi_data[self.feature_names], aqi_data[self.target_name].values.reshape(-1, 1)

        # 数据标准化
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)
        
        # 数据划分
        train_size = int(len(self.X_scaled) * 0.8)
        if train_size == 0 or train_size == len(self.X_scaled): # 确保训练集和测试集都有数据
             raise ValueError("数据量太小，无法进行训练集和测试集划分。")

        self.X_train, self.X_test = self.X_scaled[:train_size], self.X_scaled[train_size:]
        self.y_train, self.y_test = self.y_scaled[:train_size], self.y_scaled[train_size:]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))

        # 保存测试集的日期，用于绘图
        self.test_dates = aqi_data['datetime'].iloc[train_size:].reset_index(drop=True)


    def build_model(self, units=50, dropout_rate=0.2, activation_func='tanh'):
        """
        创建并编译双向LSTM模型。

        参数:
        - units: LSTM层神经元数量
        - dropout_rate: Dropout层丢弃率
        - activation_func: LSTM层的激活函数 (默认为 'tanh')

        返回:
        - model: 编译后的Keras模型
        """
        inputs = Input(shape=(self.X_train.shape[1], self.X_train.shape[2]))
        x = Bidirectional(LSTM(units=units, activation=activation_func))(inputs) # 使用传入的激活函数
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train_and_evaluate(self, model):
        """
        训练模型并评估在测试集上的表现。
        假设模型在传入此方法前已经编译好。
        """
        model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
        predictions = model.predict(self.X_test)
        return self.scaler_y.inverse_transform(predictions), self.scaler_y.inverse_transform(self.y_test)

    def save_model(self, model):
        """保存训练好的模型。"""
        try:
            model.save(self.model_path)
            print(f"模型已保存为 {self.model_path}")
        except Exception as e:
            print(f"错误: 保存模型失败: {e}")


    def load_existing_model(self):
        """加载已保存的模型。"""
        if os.path.exists(self.model_path):
            print(f"加载已保存的模型：{self.model_path}")
            try:
                return load_model(self.model_path) 
            except Exception as e:
                print(f"错误: 加载模型 {self.model_path} 失败: {e}。将尝试训练新模型。")
                return None
        else:
            print(f"模型文件 {self.model_path} 未找到，需重新训练模型。")
            return None

    def tune_model(self):
        """使用贝叶斯优化来调节模型超参数。"""
        
        tuner_base_dir = tempfile.mkdtemp(prefix="aqi_tuner_")
        print(f"Keras Tuner 将使用临时目录: {tuner_base_dir}")

        try:
            def build_tuned_model(hp):
                units = hp.Int('units', min_value=32, max_value=256, step=32) # 减小 units 的上限
                dropout_rate = hp.Float('dropout', min_value=0.1, max_value=0.4, step=0.1) # 调整 dropout 范围
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='LOG', default=1e-3) # 调整学习率范围

                # 使用 'tanh' 激活函数进行调优
                model = self.build_model(units, dropout_rate, activation_func='tanh') 
                from tensorflow import keras 
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
                return model

            tuner = BayesianOptimization(
                build_tuned_model,
                objective='val_loss',
                max_trials=10, # 可以根据需要调整试验次数
                executions_per_trial=1, # 每次试验训练一次模型
                directory=tuner_base_dir, 
                project_name='aqi_optimization', 
                overwrite=True 
            )
            # 增加回调以提前停止没有改进的试验 (可选，但有助于避免长时间运行失败的试验)
            # from tensorflow.keras.callbacks import EarlyStopping
            # early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
            # tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, verbose=0, callbacks=[early_stopping])
            tuner.search(self.X_train, self.y_train, epochs=50, validation_split=0.2, verbose=0) 
            
            best_hp_list = tuner.get_best_hyperparameters(num_trials=1)
            if not best_hp_list:
                print("警告: Keras Tuner 未找到任何最佳超参数。将返回一个使用默认参数构建的新模型。")
                return self.build_model(activation_func='tanh') # 返回默认 tanh 模型

            best_hp = best_hp_list[0]
            print("找到的最佳超参数:")
            print(f"Units: {best_hp.get('units')}")
            print(f"Dropout rate: {best_hp.get('dropout')}")
            print(f"Learning rate: {best_hp.get('learning_rate')}")
            
            best_models = tuner.get_best_models(num_models=1)
            if not best_models:
                print("警告: Keras Tuner 未能构建最佳模型。将返回一个使用默认参数构建的新模型。")
                return self.build_model(activation_func='tanh') # 返回默认 tanh 模型
            
            optimized_model = best_models[0]
            return optimized_model

        except Exception as e: # 更广泛地捕获调优过程中的异常
            print(f"Keras Tuner 调优过程中发生错误: {e}")
            print("将返回一个使用默认参数构建的新模型。")
            return self.build_model(activation_func='tanh') # 出错时返回默认 tanh 模型
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
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        mask = ~np.isnan(y_pred) & ~np.isnan(y_true) # 同时检查 y_true 中的 NaN
        if not np.any(mask): 
            print("警告: 所有预测值或真实值均为NaN，无法计算评估指标。")
            return {
                "MAPE": float('nan'), "RMSE": float('nan'),
                "R2": float('nan'), "MAE": float('nan')
            }
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        if len(y_true_filtered) == 0: 
             print("警告: 没有有效的预测值进行评估。")
             return {"MAPE": float('nan'), "RMSE": float('nan'), "R2": float('nan'), "MAE": float('nan')}

        # 避免除以零的 MAPE
        mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / np.where(y_true_filtered == 0, 1e-8, y_true_filtered))) * 100


        return {
            "MAPE": mape, # mean_absolute_percentage_error(y_true_filtered, y_pred_filtered), # 使用自定义MAPE避免0除问题
            "RMSE": np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered)),
            "R2": r2_score(y_true_filtered, y_pred_filtered),
            "MAE": mean_absolute_error(y_true_filtered, y_pred_filtered)
        }

    def plot_results(self, y_true, y_pred_base, y_pred_opt):
        """绘制实际值与预测值的对比图。"""
        plt.figure(figsize=(14, 7))
        
        y_true_flat = y_true.flatten() # 确保 y_true 是一维的

        plt.plot(self.test_dates, y_true_flat, label='实际 AQI', color='blue')
        
        valid_base_pred_mask = ~np.isnan(y_pred_base.flatten())
        if np.any(valid_base_pred_mask):
            plt.plot(self.test_dates[valid_base_pred_mask], y_pred_base.flatten()[valid_base_pred_mask], label='基线模型预测 AQI', color='orange', alpha=0.7)
        
        valid_opt_pred_mask = ~np.isnan(y_pred_opt.flatten())
        if np.any(valid_opt_pred_mask):
            plt.plot(self.test_dates[valid_opt_pred_mask], y_pred_opt.flatten()[valid_opt_pred_mask], label='优化模型预测 AQI', color='green', alpha=0.7)
        
        if pd.api.types.is_datetime64_any_dtype(self.test_dates):
             plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')) 
             plt.xticks(rotation=45)

        plt.title('AQI 预测结果对比')
        plt.xlabel('时间')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
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
            plot_data[model_type] = {k: (metrics[k] if pd.notna(metrics[k]) else 0) for k in metric_names} 

        df_metrics = pd.DataFrame(plot_data).T[metric_names] 

        ax = df_metrics.plot(kind='bar', figsize=(12, 7), colormap='viridis') 
        
        plt.title('模型评价指标对比 (MAPE, RMSE, MAE越低越好, R2越高越好)') 
        plt.ylabel('指标值')
        plt.xticks(rotation=0) 
        plt.legend(title='模型')
        plt.grid(axis='y', linestyle='--')
        
        # 在柱状图上添加数值标签的改进逻辑
        for i, model_name in enumerate(df_metrics.index): # Baseline, Optimized
            for j, metric_name in enumerate(df_metrics.columns): # MAPE, RMSE, ...
                original_value = metrics_data[model_name][metric_name] 
                display_value_text = f"{original_value:.3f}" if pd.notna(original_value) else "N/A"
                
                # 获取当前柱子的位置和高度
                bar = ax.patches[i * len(df_metrics.columns) + j]
                bar_height = bar.get_height()
                bar_x = bar.get_x() + bar.get_width() / 2.0


                # 调整文本的Y位置，使其在柱子上方
                text_y_position = bar_height + 0.01 * df_metrics[metric_name].abs().max() if pd.notna(bar_height) and bar_height >= 0 else 0.01
                if pd.isna(bar_height) or bar_height < 0 : # 处理负值或NaN高度
                     text_y_position = 0.01 # 放在靠近0的位置


                plt.text(
                    bar_x, 
                    text_y_position,
                    display_value_text, 
                    ha='center',
                    va='bottom',
                    fontsize=8 
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

def select_model_file_path(title="请选择模型文件路径（用于加载或保存）", default_name="trained_model.h5"):
    """打开文件对话框选择模型文件路径。"""
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.asksaveasfilename(
        title=title,
        initialfile=default_name,
        defaultextension=".h5",
        filetypes=[("HDF5 模型文件", "*.h5"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def main(): 
    data_path = select_data_file_path("请选择 AQI 数据集 Excel 文件")
    if not data_path:
        print("未选择数据集文件，程序退出。")
        return

    model_path_selected = select_model_file_path("请指定模型文件的加载/保存路径", "trained_model.h5")
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
        baseline_model = predictor.build_model(activation_func='tanh') # 使用 tanh
    else:
        print("重新编译已加载的基线模型以用于训练/评估...")
        baseline_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    print("训练和评估基线模型...")
    baseline_predictions, y_test_inv = predictor.train_and_evaluate(baseline_model)
    predictor.save_model(baseline_model)

    print("\n--- 优化模型 (使用贝叶斯优化) ---")
    optimized_model = predictor.tune_model() 
    
    if optimized_model is None: 
        print("警告: 模型优化失败或未执行，将使用默认参数的 tanh 模型进行评估。")
        optimized_model = predictor.build_model(activation_func='tanh') # 如果优化失败，构建一个默认的tanh模型
        # 确保这个模型也被编译
        if not optimized_model.optimizer:
            optimized_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    print("训练和评估优化后的模型...")
    # 确保优化后的模型（即使是回退的默认模型）已编译
    if not optimized_model.optimizer:
        print("编译优化（或回退的默认）模型...")
        optimized_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    optimized_predictions, _ = predictor.train_and_evaluate(optimized_model)


    print("\n--- 评估结果 ---")
    eval_baseline = AQIPredictor.evaluate(y_test_inv, baseline_predictions)
    eval_optimized = AQIPredictor.evaluate(y_test_inv, optimized_predictions) # 现在 optimized_model 总会有一个值


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
