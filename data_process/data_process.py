import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import os
import glob

# --- 配置参数 ---
DEFAULT_TARGET_CITY = "南京" # 默认目标城市
# 需要提取的空气质量指标列表
TARGET_METRICS = [
    "AQI", "PM2.5", "PM2.5_24h", "PM10", "PM10_24h",
    "SO2", "SO2_24h", "NO2", "NO2_24h", "O3", "O3_24h",
    "O3_8h", "O3_8h_24h", "CO", "CO_24h"
]
# CSV文件名的匹配模式
FILE_PATTERN = "china_cities_*.csv"
# 输出Excel工作表的名称
OUTPUT_SHEET_NAME_TEMPLATE = "{}_AQI_Data" # 使用模板，城市名会填充进来

def select_folder_path(title="请选择包含CSV文件的文件夹"):
    """
    打开一个对话框让用户选择文件夹路径。
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def select_save_file_path(title="请选择保存Excel文件的位置和名称", default_name="City_AQI_Data.xlsx"):
    """
    打开一个对话框让用户选择保存文件的路径和名称。
    """
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    file_path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".xlsx",
        initialfile=default_name,
        filetypes=[("Excel 文件", "*.xlsx"), ("所有文件", "*.*")]
    )
    root.destroy()
    return file_path

def ask_for_city_name(default_city):
    """
    弹出一个对话框让用户输入城市名称。
    """
    root = tk.Tk()
    root.withdraw() # 隐藏主窗口
    city_name = simpledialog.askstring("输入城市", f"请输入目标城市名称（默认为 {default_city}）：", initialvalue=default_city)
    root.destroy()
    if city_name is None or city_name.strip() == "": # 用户取消或输入为空
        return default_city
    return city_name.strip()

def process_air_quality_data(folder_path, output_file_path, target_city_name):
    """
    处理指定文件夹及其子文件夹中的空气质量数据并保存到Excel文件。
    """
    # 构建递归搜索模式以包含子文件夹
    search_pattern_recursive = os.path.join(folder_path, '**', FILE_PATTERN)
    
    csv_files = glob.glob(search_pattern_recursive, recursive=True)

    if not csv_files:
        messagebox.showinfo("提示", f"在文件夹 {folder_path} 及其子文件夹中未找到匹配 '{FILE_PATTERN}' 模式的文件。")
        return

    all_city_data_list = []
    # 动态构建所需列，包含用户输入的目标城市
    required_csv_columns = ['date', 'hour', 'type', target_city_name]

    print(f"目标城市: {target_city_name}")
    print(f"找到 {len(csv_files)} 个匹配文件。开始处理...")

    for file_path in csv_files:
        relative_file_path = os.path.relpath(file_path, folder_path)
        print(f"  正在处理文件: {relative_file_path}")
        try:
            df_raw = pd.read_csv(
                file_path,
                usecols=lambda col: col in required_csv_columns,
                dtype={'date': str, 'hour': object}
            )

            if target_city_name not in df_raw.columns:
                print(f"    警告: 文件 {relative_file_path} 中缺少 '{target_city_name}' 列，已跳过。")
                continue
            
            try:
                df_raw['hour'] = pd.to_numeric(df_raw['hour'], errors='coerce')
                df_raw.dropna(subset=['hour'], inplace=True)
                df_raw['hour'] = df_raw['hour'].astype(int)
            except Exception as e:
                print(f"    警告: 文件 {relative_file_path} 中 'hour' 列转换失败: {e}，已跳过此文件。")
                continue

            df_metrics_filtered = df_raw[df_raw['type'].isin(TARGET_METRICS)].copy()

            if df_metrics_filtered.empty:
                print(f"    文件 {relative_file_path} 中未找到 '{target_city_name}' 的相关指标数据。")
                continue

            df_pivoted = df_metrics_filtered.pivot_table(
                index=['date', 'hour'],
                columns='type',
                values=target_city_name # 使用传入的城市名
            ).reset_index()

            for metric in TARGET_METRICS:
                if metric not in df_pivoted.columns:
                    df_pivoted[metric] = pd.NA

            final_column_order = ['date', 'hour'] + [m for m in TARGET_METRICS if m in df_pivoted.columns]
            df_pivoted = df_pivoted[final_column_order]
            
            if not df_pivoted.empty:
                all_city_data_list.append(df_pivoted)
            print(f"    文件 {relative_file_path} 处理完毕。提取到 {len(df_pivoted)} 条记录。")

        except FileNotFoundError:
            print(f"    错误: 文件 {relative_file_path} 未找到 (glob 后)。")
        except pd.errors.EmptyDataError:
            print(f"    警告: 文件 {relative_file_path} 为空，已跳过。")
        except KeyError as e:
            # 更具体的KeyError处理，可能是目标城市列在usecols后仍不存在
            if str(e) == f"\"['{target_city_name}'] not in index\"":
                 print(f"    错误: 文件 {relative_file_path} 缺少必需的列 '{target_city_name}' (在usecols筛选后)。已跳过。")
            else:
                print(f"    错误: 文件 {relative_file_path} 缺少必需的列: {e}。已跳过。")
        except Exception as e:
            print(f"    处理文件 {relative_file_path} 时发生未知错误: {e}。已跳过。")

    if not all_city_data_list:
        messagebox.showinfo("提示", f"未能从任何文件中提取到 '{target_city_name}' 的有效数据。")
        return

    processed_data_frames = [df for df in all_city_data_list if not df.empty and not df.drop(columns=['date', 'hour'], errors='ignore').isnull().all().all()]

    if not processed_data_frames:
        messagebox.showinfo("提示", f"所有为 '{target_city_name}' 提取的数据均为空或无效，未能生成最终文件。")
        return
        
    final_df = pd.concat(processed_data_frames, ignore_index=True)

    final_df.sort_values(by=['date', 'hour'], inplace=True)
    final_df.drop_duplicates(subset=['date', 'hour'], keep='first', inplace=True)

    # 根据城市名生成工作表名
    sheet_name = OUTPUT_SHEET_NAME_TEMPLATE.format(target_city_name)

    try:
        final_df.to_excel(output_file_path, sheet_name=sheet_name, index=False)
        messagebox.showinfo("成功", f"'{target_city_name}' 的数据已成功提取并保存到:\n{output_file_path}\n工作表: {sheet_name}")
        print(f"\n'{target_city_name}' 的数据已成功提取并保存到: {output_file_path} (工作表: {sheet_name})")
    except PermissionError as e_perm:
        error_message = (
            f"保存Excel文件失败：权限不足。\n"
            f"错误详情: {e_perm}\n\n"
            f"请尝试以下操作：\n"
            f"1. 检查文件 '{os.path.basename(output_file_path)}' 是否已在其他程序中打开，如果是，请关闭它。\n"
            f"2. 尝试将文件保存到您确定具有写入权限的其他位置（例如“我的文档”文件夹）。\n"
            f"3. 确保您有权限写入到目标文件夹 '{os.path.dirname(output_file_path)}'。"
        )
        messagebox.showerror("保存失败 - 权限错误", error_message)
        print(f"\n保存Excel文件失败: {e_perm}")
    except Exception as e:
        messagebox.showerror("保存失败", f"保存Excel文件失败: {e}")
        print(f"\n保存Excel文件失败: {e}")

if __name__ == "__main__":
    # 0. 让用户输入目标城市
    target_city_input = ask_for_city_name(DEFAULT_TARGET_CITY)
    print(f"选择的目标城市: {target_city_input}")

    # 1. 让用户选择输入文件夹
    input_folder = select_folder_path()

    if not input_folder:
        print("未选择输入文件夹，操作已取消。")
    else:
        print(f"选择的输入文件夹: {input_folder}")
        
        # 2. 让用户选择输出文件路径和名称，文件名可以包含城市名
        default_output_filename = f"{target_city_input}_AQI_Data.xlsx"
        output_excel_file = select_save_file_path(default_name=default_output_filename)

        if not output_excel_file:
            print("未选择输出文件，操作已取消。")
        else:
            print(f"选择的输出文件: {output_excel_file}")
            # 3. 处理数据并保存
            process_air_quality_data(input_folder, output_excel_file, target_city_input)
