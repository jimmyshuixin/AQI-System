# **AQI预测与异常检测系统 (AQISystem\_v3.2)**
# **本项目为本人在辅修“机器学习”的学习成果，内有许多不足之处，希望有大佬批评指正！感谢！**
## **目录**

1. [项目概述](#bookmark=id.je0v2mfjzjtk)  
2. [核心功能](#bookmark=id.1ytq6t1l1qhj)  
3. [系统组件](#bookmark=id.eeh6w7nxtagg)  
   * [数据ETL与高级预处理](#bookmark=id.dpvrj2poojsh)  
   * [模型训练 (ModelTrainer)](#bookmark=id.4f8plbd390db)  
   * [模型评估 (ModelEvaluator)](#bookmark=id.vlxq3k7fgtcn)  
   * [模型预测 (ModelPredictor)](#bookmark=id.lw7bin9s6qbc)  
   * [异常数据检测](#bookmark=id.ls8l4etq22m)  
   * [主控系统 (AQISystem)](#bookmark=id.eqm6lcqtk4tm)  
4. [技术栈](#bookmark=id.ry8p0k1iaqmu)  
5. [文件结构](#bookmark=id.3mzhx7shr0l9)  
6. [安装与环境配置](#bookmark=id.7ikoc7jjgbff)  
7. [核心参数与配置](#bookmark=id.hb1lno7oey18)  
8. [使用说明](#bookmark=id.fgh81mid7gef)  
   * [1\. 训练新模型](#bookmark=id.g1wh81slh5rg)  
   * [2\. 评估现有模型性能](#bookmark=id.e5t94ko4iozi)  
   * [3\. 使用现有模型进行预测](#bookmark=id.diqa2wclbn7i)  
   * [4\. 使用现有模型检测异常数据](#bookmark=id.s7ekik13rdlj)  
9. [日志系统](#bookmark=id.gsjgs28dm1wb)  
10. [输出说明](#bookmark=id.ubyhammceoeq)  
11. [注意事项与未来工作](#bookmark=id.1rmx8588dxgf)

## **项目概述**

AQISystem\_v3.2.py 脚本构建了一个综合的空气质量指数（AQI）及多种污染物浓度预测与异常数据检测系统。该系统利用基于Transformer的深度学习模型进行多目标时间序列预测，并集成了数据预处理、模型训练、超参数优化、模型评估、预测以及异常点识别等功能。系统通过命令行界面与用户交互，并将所有控制台输出及详细的训练过程记录到日志文件中。

## **核心功能**

* **多目标空气质量预测**：利用Transformer模型同步预测AQI及PM2.5, PM10, SO2, NO2, O3, CO等多种污染物未来一段时间的浓度。  
* **高级数据预处理**：包括精确的时间戳处理、鲁棒的数值转换、全面的缺失值填充、基于IQR的异常值检测与平滑（训练时可选）。  
* **高级特征工程**：  
  * 依据国家标准（GB 3095-2012）动态计算AQI作为输入特征。  
  * 生成时间周期性特征（小时、星期、月份的正余弦编码）。  
  * 为目标变量及关键污染物浓度（如24小时、8小时平均浓度）构建滞后特征。  
* **自动化模型训练与优化**：  
  * 使用PyTorch实现的Transformer模型。  
  * 集成Optuna进行超参数自动搜索与优化。  
  * 应用学习率动态调整（ReduceLROnPlateau）和早停（Early Stopping）机制。  
* **独立的模型评估模块**：计算MAE, RMSE, R², MAPE, SMAPE等多种指标，并生成实际值与预测值的对比图。  
* **便捷的模型预测功能**：加载预训练模型对新数据进行预测，并可将结果保存为CSV。  
* **基于模型的异常数据检测**：利用训练好的模型识别输入数据序列中的潜在异常点，并可视化。  
* **全面的日志记录**：所有控制台输出自动保存到主日志文件，训练过程有独立的详细日志。  
* **命令行交互界面**：引导用户选择执行不同功能。

## **系统组件**

### **数据ETL与高级预处理**

此部分负责数据的加载、清洗、转换和特征工程。关键步骤包括：

* **时间戳解析**：从输入数据（CSV/Excel）中解析日期和小时信息，构建统一的datetime索引。  
* **数值转换与缺失值处理**：将所有数据列尝试转换为数值类型，对无法转换的值设为NaN。特征列的NaN值通过前向填充(ffill)、后向填充(bfill)及0填充处理。目标列的NaN值在训练时会导致对应行被删除，或在异常检测时被填充。  
* **IQR异常值检测与插值** (训练时可选)：对目标列，若启用，则使用IQR方法识别统计学上的异常值，将其替换为NaN，然后使用时间插值（或线性插值）进行平滑。  
* **AQI计算** (calculate\_aqi\_from\_pollutants)：根据GB 3095-2012标准，利用PM2.5\_24h, PM10\_24h, SO2\_24h, NO2\_24h, CO\_24h, O3\_8h\_24h (或其小时近似值) 计算每条记录的AQI\_calculated和Primary\_Pollutant\_calculated。AQI\_calculated可作为模型输入特征。  
* **周期性特征生成**：为时间索引生成hour\_sin, hour\_cos, dayofweek\_sin, dayofweek\_cos, month\_sin, month\_cos等特征，帮助模型捕捉时间模式。  
* **滞后特征生成**：为所有目标列以及数据中存在的\_24h或\_8h结尾的列（如PM2.5\_24h）创建多阶滞后特征（如PM2.5\_lag\_1, AQI\_lag\_1等），引入历史依赖性。滞后阶数根据回溯窗口大小动态确定。  
* **数据标准化**：使用sklearn.preprocessing.StandardScaler对所有最终选定的数值输入特征进行标准化。对每个目标变量也使用独立的StandardScaler进行标准化。这些缩放器在训练时拟合和保存，在预测和评估时加载使用。

### **模型训练 (ModelTrainer)**

ModelTrainer类负责整个模型训练流程：

* **数据加载与预处理调用**：调用核心预处理逻辑 (\_load\_and\_preprocess\_data\_core)，并在此阶段拟合和保存特征及目标缩放器。  
* **序列创建** (create\_sequences)：将预处理后的时间序列数据转换为适用于监督学习的(X, y)样本对，其中X是长度为look\_back的输入序列，y是长度为horizon的目标序列。  
* **数据集划分**：将序列数据划分为训练集、验证集和（内部）测试集。  
* **超参数优化 (Optuna)** (\_objective\_optuna)：  
  * 定义搜索空间，包括学习率、Transformer模型维度 (d\_model)、注意力头数 (nhead)、编码器层数 (num\_encoder\_layers)、前馈网络维度因子、Dropout率、LayerNorm位置 (norm\_first)、权重衰减以及学习率调度器参数等。  
  * 对每组超参数，构建模型并进行指定轮数 (optuna\_epochs) 的训练与验证，使用验证集上主要目标（如AQI）的损失作为优化目标。  
  * 使用HyperbandPruner进行剪枝，提前终止不良试验。  
* **最终模型训练** (\_train\_model\_core)：  
  * 使用Optuna找到的最佳超参数构建AQITransformer模型。  
  * 使用训练集进行训练，验证集进行性能监控。  
  * 采用均方误差（MSE）作为损失函数。  
  * 使用AdamW优化器。  
  * 应用ReduceLROnPlateau学习率调度器（基于验证集上主要目标的损失）。  
  * 应用早停机制（基于验证集上主要目标的损失，若连续多轮未改善则停止）。  
  * 保存训练过程中验证损失最佳的模型状态 (best\_aqi\_transformer\_model\_adv.pth)。  
* **保存工件**：保存最佳模型权重、最终模型配置（包括架构参数、特征列表等，存为model\_config\_adv.json）、特征缩放器 (aqi\_feature\_scaler\_adv.pkl) 和目标缩放器 (aqi\_target\_scalers\_adv.pkl)。  
* **日志与可视化**：记录详细训练日志到单独文件，并绘制训练/验证损失曲线图 (final\_model\_training\_loss.png)。

### **模型评估 (ModelEvaluator)**

ModelEvaluator类用于评估已训练模型的性能：

* **加载模型与组件**：需要传入已训练的模型实例、模型配置、特征缩放器和目标缩放器。  
* **数据预处理** (\_preprocess\_evaluation\_data)：对用户提供的测试数据（文件或DataFrame）执行与训练时一致的特征工程和特征缩放。目标值保持原始尺度。  
* **序列创建**：将预处理后的评估数据转换为(X, y)序列。  
* **模型预测**：使用加载的模型对X序列进行预测，得到缩放后的预测值。  
* **反向缩放**：使用保存的目标缩放器将预测值反向转换回原始尺度。  
* **指标计算**：对每个目标变量，计算以下评估指标：  
  * 平均绝对误差 (MAE)  
  * 均方根误差 (RMSE)  
  * R平方 (R²)  
  * 平均绝对百分比误差 (MAPE)  
  * 对称平均绝对百分比误差 (SMAPE)  
* **结果可视化**：为每个目标变量绘制实际值与预测值的对比图，并保存。  
* **两种评估入口**：  
  * evaluate\_model\_from\_source: 从原始数据文件或DataFrame开始评估。  
  * evaluate\_from\_prepared\_sequences: 直接使用已创建好的X, y序列进行评估（例如，训练流程内部调用）。

### **模型预测 (ModelPredictor)**

ModelPredictor类用于使用已训练模型进行空气质量预测：

* **加载工件** (\_load\_artifacts)：自动从指定目录加载模型配置 (model\_config\_adv.json)、模型权重 (best\_aqi\_transformer\_model\_adv.pth)、特征缩放器 (aqi\_feature\_scaler\_adv.pkl) 和目标缩放器 (aqi\_target\_scalers\_adv.pkl)。  
* **输入数据预处理** (\_preprocess\_input\_for\_prediction)：对用户提供的输入数据（文件或DataFrame，通常是历史数据）执行与训练时一致的特征工程和特征缩放。输入数据长度至少应为look\_back。  
* **序列准备**：从预处理后的数据中截取最后look\_back个时间步作为模型输入序列。  
* **模型预测**：将输入序列传递给加载的AQITransformer模型，得到未来horizon个时间步的缩放后预测值。  
* **反向缩放**：使用加载的目标缩放器将预测结果转换回原始污染物浓度/指数值。  
* **结果输出**：  
  * 在控制台打印未来几个小时的预测摘要。  
  * 如果能确定输入数据的最后一个时间戳，则可选择将详细的、带未来时间戳的预测结果（所有目标变量，未来horizon小时）保存到predictions\_output\_adv.csv文件中。

### **异常数据检测**

此功能集成在AQISystem类的detect\_anomalies方法中：

* **加载模型与组件**：确保已训练的模型及其相关配置、缩放器已加载。  
* **数据预处理** (\_preprocess\_input\_for\_anomaly)：对输入数据（包含实际观测值）进行特征工程和特征缩放。目标列的NaN值会特殊处理（如填充0），因为需要与模型预测进行比较。  
* **序列创建**：将数据转换为(X, y)序列，但**关键在于horizon强制设为1**，因为异常检测基于模型对下一个时间步的重建能力。  
* **单步预测**：模型对X序列进行单步预测（预测下一个时间步的值）。  
* **计算重建误差**：将模型单步预测值反向缩放后，与真实的单步实际值进行比较，计算绝对误差。  
* **异常判定**：  
  * 对每个目标污染物，计算其重建误差序列的均值和标准差。  
  * 异常阈值定义为：均值误差 \+ threshold\_factor \* 标准差误差。threshold\_factor可由用户指定或使用默认值。  
  * 重建误差超过该阈值的点被标记为潜在异常点。  
* **报告与可视化**：  
  * 为每个目标污染物生成异常报告，包括异常点数量、使用的阈值、异常点的时间戳（如果可用）、实际值、预测值和误差。  
  * 绘制每个目标污染物的时间序列图，并在图中标出检测到的异常点，保存为图像文件。

### **主控系统 (AQISystem)**

AQISystem类是整个系统的入口和协调者：

* **配置管理**：加载默认配置，并允许通过config\_overrides进行参数定制。  
* **统一模型加载** (\_ensure\_model\_loaded\_for\_use)：集中管理模型、配置和缩放器的加载，供预测、评估、异常检测等功能共享。  
* **功能调度**：根据用户在命令行界面的选择，调用相应的方法：  
  * train\_new\_model(): 初始化ModelTrainer并执行训练流程。  
  * predict\_with\_existing\_model(): 初始化（或复用）ModelPredictor（或直接使用内部加载的模型）执行预测。  
  * evaluate\_existing\_model(): 初始化ModelEvaluator执行评估。  
  * detect\_anomalies(): 执行异常检测逻辑。

## **技术栈**

* **Python 3.x**  
* **Pandas**: 数据处理与分析。  
* **NumPy**: 数值计算。  
* **Matplotlib**: 数据可视化与绘图。  
* **Scikit-learn**: 特征标准化 (StandardScaler) 和模型评估指标。  
* **PyTorch**: 深度学习框架，用于实现Transformer模型。  
* **Optuna**: 超参数优化框架。  
* **Joblib**: 保存和加载Python对象（如Scikit-learn的缩放器）。  
* **JSON**: 处理模型配置文件。  
* **Logging**: Python内置日志模块。  
* **Datetime**: 处理日期和时间。

## **文件结构**

建议的项目文件结构如下：

AQI\_Prediction\_System/  
│  
├── AQISystem\_v3.2.py               \# 主脚本文件  
│  
├── data\_process/                   \# 数据处理相关（脚本假设，实际数据可放此处）  
│   └── output/  
│       └── 南京\_AQI\_Data.xlsx        \# 默认示例数据文件 (或用户指定路径)  
│  
├── model/                          \# 模型相关文件  
│   └── output/                     \# 模型训练产物保存目录 (可配置)  
│       ├── best\_aqi\_transformer\_model\_adv.pth  \# 最佳模型权重  
│       ├── aqi\_feature\_scaler\_adv.pkl     \# 特征缩放器  
│       ├── aqi\_target\_scalers\_adv.pkl     \# 目标缩放器字典  
│       ├── model\_config\_adv.json          \# 模型配置信息  
│       ├── final\_model\_training\_loss.png  \# 最终模型训练损失图  
│       ├── evaluation\_\*.png               \# 评估时的预测对比图  
│       ├── anomaly\_detection\_report\_\*.png \# 异常检测图  
│       ├── predictions\_output\_adv.csv     \# 预测结果CSV (如果用户选择保存)  
│       └── logs/                          \# 日志文件目录  
│           ├── console\_output\_YYYYMMDD\_HHMMSS.log  \# 主控制台日志  
│           └── training\_specific\_log\_YYYYMMDD\_HHMMSS.log \# 特定训练任务的日志  
│  
└── README.md                       \# 本说明文件

**注意**:

* data\_process/output/ 和 model/output/ 目录及其子目录（如 logs/）如果不存在，脚本会尝试创建它们。  
* 用户可以在运行时通过命令行提示指定数据文件路径和模型工件保存目录。

## **安装与环境配置**

1. **Python环境**: 确保已安装Python 3.7或更高版本。建议使用虚拟环境（如venv或conda）。  
   python \-m venv aqi\_env  
   source aqi\_env/bin/activate  \# Linux/macOS  
   \# aqi\_env\\Scripts\\activate   \# Windows

2. **安装依赖库**:  
   pip install scikit-learn keras keras_tuner pandas numpy matplotlib tensorflow openpyxl optuna
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

   * openpyxl 用于读写Excel文件。  
   * PyTorch的安装可能需要根据您的操作系统和CUDA版本选择特定命令，请参考[PyTorch官网](https://pytorch.org/get-started/locally/)。如果不需要GPU支持，CPU版本即可。  
3. 中文字体 (可选，用于Matplotlib绘图):  
   脚本尝试设置SimHei作为Matplotlib的默认中文字体。如果系统中没有该字体，图表中的中文可能无法正常显示。可以安装SimHei字体，或在脚本中修改plt.rcParams\['font.sans-serif'\]为系统中已有的支持中文的字体。

## **核心参数与配置**

脚本中定义了许多全局参数，部分可以通过修改脚本顶部的常量进行调整，部分与模型训练和保存相关。

* **数据与模型路径**:  
  * DEFAULT\_FILE\_PATH: 默认的输入数据文件路径。  
  * MODEL\_ARTIFACTS\_DIR: 模型相关文件（权重、配置、缩放器等）的保存和加载目录。  
  * MODEL\_STATE\_SAVE\_NAME, FEATURE\_SCALER\_SAVE\_NAME, TARGET\_SCALERS\_SAVE\_NAME, MODEL\_CONFIG\_SAVE\_NAME: 各类工件的默认文件名。  
* **时间序列参数**:  
  * DEFAULT\_LOOK\_BACK: 模型回溯窗口大小（用过去多少小时数据预测）。  
  * DEFAULT\_HORIZON: 模型预测范围（预测未来多少小时）。  
* **目标变量**:  
  * DEFAULT\_TARGET\_COL\_NAMES: 需要预测的目标列名列表 (e.g., \['AQI', 'PM2.5', ...\])。  
  * DEFAULT\_PRIMARY\_TARGET\_COL\_NAME: 主要目标变量，用于早停和学习率调整 (e.g., 'AQI')。  
* **训练参数**:  
  * DEFAULT\_BATCH\_SIZE: 训练时的批处理大小。  
  * DEFAULT\_FULL\_TRAIN\_EPOCHS: 最终模型训练时的最大轮数。  
  * DEFAULT\_N\_OPTUNA\_TRIALS: Optuna超参数优化时的试验次数。  
  * DEFAULT\_OPTUNA\_EPOCHS: Optuna每次试验训练的轮数。  
  * DEFAULT\_EARLY\_STOPPING\_PATIENCE: 早停机制的耐心值。  
  * DEFAULT\_MIN\_DELTA: 判断验证损失是否有改善的最小变化量。  
  * DEFAULT\_ENABLE\_IQR\_OUTLIER\_DETECTION: 是否在训练预处理中启用基于IQR的异常值检测（默认为False）。  
* **异常检测参数**:  
  * DEFAULT\_ANOMALY\_THRESHOLD\_FACTOR: 异常检测时，判断异常的阈值因子。  
* **AQI计算标准**:  
  * IAQI\_LEVELS, POLLUTANT\_BREAKPOINTS, POLLUTANT\_BREAKPOINTS\_HOURLY\_APPROX: 用于计算AQI的国家标准浓度限值。

大部分路径和部分关键参数（如IQR启用状态、异常检测阈值因子）可以在脚本运行时通过命令行提示进行覆盖。

## **使用说明**

直接运行Python脚本 AQISystem\_v3.2.py：

python AQISystem\_v3.2.py

脚本启动后，会显示一个命令行菜单，引导用户选择要执行的操作：

您好！请选择要执行的操作:  
 (1: 训练新模型)  
 (2: 评估现有模型性能)  
 (3: 使用现有模型进行预测)  
 (4: 使用现有模型检测异常数据)  
请输入选项 (1-4):

根据提示输入相应的数字并按回车。

### **1\. 训练新模型**

* **提示**:  
  * 请输入模型文件及相关组件的保存目录 (默认为 'model\\output'): 指定训练产物（模型、配置、缩放器、日志等）的保存位置。  
  * 请输入训练数据文件的完整路径 (默认为 'data\_process\\output\\南京\_AQI\_Data.xlsx'): 提供包含历史空气质量数据的CSV或Excel文件路径。  
  * 是否在训练预处理中启用IQR异常值检测? (y/n, 默认为 n): 选择是否在数据预处理阶段对目标变量进行IQR异常值检测与插值平滑。  
* **流程**:  
  1. 加载并预处理数据（包括特征工程、缩放器拟合与保存）。  
  2. 进行Optuna超参数优化，搜索最佳模型架构和训练参数。  
  3. 使用最佳超参数训练最终的Transformer模型，并保存模型权重、配置信息及缩放器。  
  4. （可选）在内部测试集上评估新训练的模型。  
* **输出**: 模型文件、配置文件、缩放器文件、训练日志、损失曲线图等会保存在指定的模型工件目录中。

### **2\. 评估现有模型性能**

* **提示**:  
  * 请输入已训练模型文件所在目录 (默认为 'model\\output'): 指定之前训练好的模型及其相关文件所在的目录。  
  * 请输入用于评估模型的数据集路径 (CSV 或 Excel格式): 提供一个独立的测试数据集（应包含与训练数据相似的列，包括真实的目标值）。  
* **流程**:  
  1. 从指定目录加载模型配置、权重和缩放器。  
  2. 加载并预处理用户提供的评估数据。  
  3. 使用模型对评估数据进行预测。  
  4. 计算各项评估指标（MAE, RMSE, R², MAPE, SMAPE）并打印。  
  5. 生成并保存实际值与预测值的对比图。  
* **输出**: 控制台打印评估指标，并在模型工件目录中保存评估图表。

### **3\. 使用现有模型进行预测**

* **提示**:  
  * 请输入已训练模型文件所在目录 (默认为 'model\\output'): 指定之前训练好的模型及其相关文件所在的目录。  
  * 请输入用于预测的输入数据文件路径 (CSV 或 Excel格式，选择训练该模型的数据集): 提供用于预测的历史数据。该数据文件应包含至少look\_back（模型回溯窗口大小，通常为72）条连续的历史记录，并且其列结构应与训练模型时的数据一致。脚本会使用此数据的最后look\_back条记录作为模型输入。  
* **流程**:  
  1. 从指定目录加载模型配置、权重和缩放器。  
  2. 加载并预处理用户提供的输入数据。  
  3. 使用模型的最后look\_back条数据进行未来horizon（预测范围，通常为72）小时的预测。  
  4. 打印预测摘要，并询问是否将详细预测结果保存到CSV文件 (predictions\_output\_adv.csv)。  
* **输出**: 控制台打印预测摘要。如果选择保存，则在模型工件目录中生成带时间戳的预测CSV文件。

### **4\. 使用现有模型检测异常数据**

* **提示**:  
  * 请输入已训练模型文件所在目录 (默认为 'model\\output'): 指定之前训练好的模型及其相关文件所在的目录。  
  * 请输入用于异常检测的数据文件路径 (CSV 或 Excel格式，模型预测值对应的真实值文件，如“南京\_AQI\_Data\_test.xlsx”): 提供包含实际观测值的时间序列数据。  
  * 请输入异常检测的阈值敏感度因子 (默认为 3.0): 自定义异常判定的敏感度。值越大，判定越宽松（需要更大的误差才算异常）。  
* **流程**:  
  1. 从指定目录加载模型配置、权重和缩放器。  
  2. 加载并预处理用户提供的待检测数据。  
  3. 模型对数据进行单步预测，计算预测值与实际值之间的重建误差。  
  4. 根据重建误差的统计特性和用户指定的阈值因子，识别潜在异常点。  
  5. 为每个目标污染物生成异常报告，包括异常点数量、时间戳、实际值、预测值等。  
  6. 生成并保存标记了异常点的时间序列图。  
* **输出**: 控制台打印异常检测报告摘要，并在模型工件目录中保存详细的异常点图表。

## **日志系统**

系统集成了全面的日志记录功能：

* **全局控制台日志** (console\_output\_YYYYMMDD\_HHMMSS.log):  
  * 所有输出到控制台的信息（包括print语句和标准错误输出）都会被捕获并实时保存到位于模型工件目录的logs子目录下的主日志文件中。  
  * 文件名包含时间戳，每次运行会生成新的日志文件。  
* **训练特定日志** (training\_specific\_log\_YYYYMMDD\_HHMMSS.log):  
  * 当执行“训练新模型”操作时，会额外创建一个详细记录该次训练过程的日志文件，同样保存在logs子目录。  
  * 此日志包含更详细的训练步骤信息、Optuna优化过程、模型参数等。  
* **日志级别**:  
  * 控制台输出级别默认为INFO。  
  * 文件记录级别默认为DEBUG，以捕获更详细的信息。  
* **配置**: 日志格式、级别等通过setup\_global\_logging\_and\_redirect和get\_configured\_logger函数进行配置。

## **输出说明**

根据执行的操作，系统会在指定的模型工件目录（默认为model/output/）下生成以下类型的文件：

* **模型文件**:  
  * best\_aqi\_transformer\_model\_adv.pth: PyTorch模型权重。  
* **配置文件**:  
  * model\_config\_adv.json: JSON格式的模型配置，包含模型架构、训练参数、特征列表等。  
* **缩放器**:  
  * aqi\_feature\_scaler\_adv.pkl: 用于输入特征的Scikit-learn StandardScaler对象。  
  * aqi\_target\_scalers\_adv.pkl: 包含各目标变量的StandardScaler对象的字典。  
* **日志文件** (位于 logs/ 子目录):  
  * console\_output\_\*.log: 主控制台日志。  
  * training\_specific\_log\_\*.log: 训练任务的详细日志。  
* **图表文件**:  
  * final\_model\_training\_loss.png: 最终模型训练过程中的训练和验证损失曲线。  
  * evaluation\_prepared\_seq\_{target\_name}\_predictions.png: 模型评估时，各目标实际值与预测值的对比图。  
  * anomaly\_detection\_report\_{target\_name}\_anomalies.png: 异常检测时，各目标时间序列中标注异常点的图。  
* **预测结果CSV** (如果用户选择保存):  
  * predictions\_output\_adv.csv: 包含未来horizon小时各目标预测值的CSV文件，带有日期和小时列。

## **注意事项与未来工作**

* **数据质量**: 模型的性能高度依赖于输入数据的质量和数量。请确保提供的数据准确、完整，并有足够长的时间跨度。  
* **计算资源**: 训练深度学习模型（尤其是使用Optuna进行超参数优化时）可能需要较长时间和较多计算资源（CPU/GPU）。  
* **依赖版本**: 确保所有依赖库的版本兼容。  
* **路径问题**: 请确保在命令行输入文件路径时使用正确的格式（例如，Windows下可能需要注意反斜杠\\）。  
* **可扩展性**:  
  * 可以考虑将数据加载、预处理、模型定义等模块进一步解耦，方便替换或扩展。  
  * 支持更多类型的模型或超参数优化策略。  
  * 开发更完善的图形用户界面（GUI）而非仅命令行。  
* **错误处理**: 脚本包含基本的错误处理，但更复杂的场景可能需要更细致的异常管理。  
* **AQI计算细节**: AQI的计算依赖于特定时间平均的污染物浓度（如PM2.5的24小时平均）。如果输入数据中缺少这些精确的平均值，脚本会尝试使用小时浓度值进行近似计算，这可能影响AQI\_calculated特征的准确性。
