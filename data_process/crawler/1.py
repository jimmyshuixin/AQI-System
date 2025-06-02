import requests
import pandas as pd
import time
import os
from datetime import datetime
import re # 用于从字符串中提取数字
import json # For parsing JSON

# --- 配置 ---
BASE_URL = "https://air.cnemc.cn:18007"
API_ENDPOINT = "/HourChangesPublish/GetAllAQIPublishLive" # New API endpoint

# 更新请求头，参考用户提供的脚本
HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
    'Connection': 'keep-alive',
    'Origin': 'https://air.cnemc.cn:18007',
    'Referer': 'https://air.cnemc.cn:18007/',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
    'X-Requested-With': 'XMLHttpRequest',
    'sec-ch-ua': '"Microsoft Edge";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
    'sec-ch-ua-mobile': '?0',
    # Content-Type might be needed for POST, but reference script doesn't show form data, so requests might set it automatically or it's not strictly needed.
    # 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8', # Example if it were a form post
}

# Excel列名顺序 (与用户原始需求一致)
EXPECTED_COLUMNS = ["date", "hour", "AQI", "PM2.5", "PM10", "SO2", "NO2", "O3", "CO"]

# --- 辅助函数 ---

def fetch_all_stations_data(session):
    """
    从 GetAllAQIPublishLive API 获取所有站点的实时AQI数据。
    """
    url = BASE_URL + API_ENDPOINT
    max_retries = 3
    retry_delay_seconds = 10
    request_timeout_seconds = 60

    for attempt in range(max_retries):
        try:
            print(f"尝试第 {attempt + 1}/{max_retries} 次从API获取所有站点数据...")
            # 参考脚本使用 POST 请求
            response = session.post(url, headers=HEADERS, timeout=request_timeout_seconds, verify=False)
            response.raise_for_status() # 如果请求失败则抛出异常
            
            # 检查响应内容类型是否为JSON
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type.lower() or "text/json" in content_type.lower() or "text/plain" in content_type.lower() : # text/plain for some servers
                return response.json()
            else:
                print(f"错误：API响应的内容类型不是JSON，而是 '{content_type}'。")
                print(f"响应内容预览: {response.text[:500]}")
                return None

        except requests.exceptions.Timeout:
            print(f"从API获取数据超时 (尝试 {attempt + 1}/{max_retries})。")
            if attempt < max_retries - 1:
                print(f"{retry_delay_seconds}秒后重试...")
                time.sleep(retry_delay_seconds)
            else:
                print("从API获取数据多次超时，放弃。")
                return None
        except requests.exceptions.RequestException as e:
            print(f"从API获取数据失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"{retry_delay_seconds}秒后重试...")
                time.sleep(retry_delay_seconds)
            else:
                print("从API获取数据多次请求失败，放弃。")
                return None
        except json.JSONDecodeError as e:
            print(f"解析API返回的JSON数据失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            print(f"收到的非JSON响应内容预览: {response.text[:500]}") # 打印部分内容帮助调试
            if attempt < max_retries - 1:
                print(f"{retry_delay_seconds}秒后重试...")
                time.sleep(retry_delay_seconds)
            else:
                print("解析API JSON数据多次失败，放弃。")
                return None
    return None # 所有重试均失败

def parse_and_filter_aqi_data(all_stations_json_data, target_city, target_station_name):
    """
    从API返回的JSON数据中解析并筛选特定城市和站点的数据。
    """
    if not all_stations_json_data:
        print("错误：传入的JSON数据为空。")
        return []

    parsed_data_list = []

    for station_data in all_stations_json_data:
        # 根据参考脚本，城市信息在 'area' 字段，站点名在 'positionname'
        # 需要确保大小写和空格匹配，或进行更宽松的匹配
        current_city = station_data.get('area', '').strip()
        current_station = station_data.get('positionname', '').strip()

        if current_city == target_city.strip() and current_station == target_station_name.strip():
            data_entry = {}
            
            # 解析时间戳: timepointstr 格式如 "2024年05月31日 22时"
            timepoint_str = station_data.get('timepointstr', '')
            date_val = None
            hour_val = None
            if timepoint_str:
                # "2024年05月31日 22时" -> "2024-05-31", 22
                match = re.search(r"(\d{4})年(\d{2})月(\d{2})日\s*(\d{1,2})时", timepoint_str)
                if match:
                    year, month, day, hour = match.groups()
                    try:
                        date_val = f"{year}-{month}-{day}"
                        hour_val = int(hour)
                    except ValueError:
                        print(f"警告：从 '{timepoint_str}' 解析日期或小时失败。")
                else:
                    print(f"警告：时间戳字符串 '{timepoint_str}' 格式不匹配。")
            
            if date_val is None or hour_val is None:
                print(f"警告：无法为站点 '{current_station}' 解析有效的时间戳 '{timepoint_str}'，跳过此记录。")
                continue

            data_entry['date'] = date_val
            data_entry['hour'] = hour_val

            def get_value_from_json(value, data_type=str):
                if isinstance(value, str) and (value == '—' or value == '-' or value.strip() == ''):
                    return None
                if value is None:
                    return None
                try:
                    if data_type == int:
                        return int(float(str(value))) # 先转float再转int，处理 "10.0" 这种
                    if data_type == float:
                        return float(str(value))
                    return str(value) # 默认返回字符串
                except (ValueError, TypeError):
                    # print(f"警告：无法将值 '{value}' 转换为 {data_type.__name__}。")
                    return None
            
            # 字段映射 (JSON key -> Our key)
            # 参考脚本中的key是小写，如 'aqi', 'pm2_5'
            data_entry['AQI']   = get_value_from_json(station_data.get('aqi'), int)
            data_entry['PM2.5'] = get_value_from_json(station_data.get('pm2_5'), float) # pm2_5 in ref script
            data_entry['PM10']  = get_value_from_json(station_data.get('pm10'), float)
            data_entry['SO2']   = get_value_from_json(station_data.get('so2'), float)
            data_entry['NO2']   = get_value_from_json(station_data.get('no2'), float)
            data_entry['O3']    = get_value_from_json(station_data.get('o3'), float)
            data_entry['CO']    = get_value_from_json(station_data.get('co'), float)
            
            # 确保所有期望的列都存在
            final_data_entry = {col: data_entry.get(col) for col in EXPECTED_COLUMNS}
            parsed_data_list.append(final_data_entry)
            
            # 假设每个站点在API响应中只有一条最新的记录
            # 如果找到目标站点，可以提前退出循环以提高效率
            break 
            
    if not parsed_data_list:
        print(f"警告：在API数据中未找到目标城市 '{target_city}' 和站点 '{target_station_name}' 的匹配记录。")
        # 可以打印一些API返回的城市和站点名称帮助调试
        sample_locations = [(d.get('area',''), d.get('positionname','')) for d in all_stations_json_data[:5]]
        print(f"API数据中的一些地点示例: {sample_locations}")

    return parsed_data_list


def update_excel_data(filepath, new_data_list):
    """
    将新数据更新或追加到Excel文件。
    如果文件存在，则读取、合并（新数据覆盖旧的同时间点数据）、排序后保存。
    """
    if not new_data_list:
        print("没有新的数据需要写入Excel。")
        return

    new_df = pd.DataFrame(new_data_list)
    if new_df.empty:
        print("新的数据DataFrame为空，不写入Excel。")
        return

    for col in EXPECTED_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None 
    new_df = new_df[EXPECTED_COLUMNS] 

    if 'hour' in new_df.columns:
        new_df['hour'] = pd.to_numeric(new_df['hour'], errors='coerce').astype('Int64')

    if os.path.exists(filepath):
        try:
            existing_df = pd.read_excel(filepath)
            if 'hour' in existing_df.columns:
                 existing_df['hour'] = pd.to_numeric(existing_df['hour'], errors='coerce').astype('Int64')
            if 'date' in existing_df.columns: 
                 existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

            new_df_indexed = new_df.set_index(['date', 'hour'], drop=False) 
            existing_df_indexed = existing_df.set_index(['date', 'hour'], drop=False)
            
            idx_to_drop = existing_df_indexed.index.intersection(new_df_indexed.index)
            existing_df_filtered = existing_df_indexed.drop(idx_to_drop)

            combined_df = pd.concat([existing_df_filtered, new_df_indexed], ignore_index=True)
            combined_df.reset_index(drop=True, inplace=True)
            
            combined_df.drop_duplicates(subset=['date', 'hour'], keep='last', inplace=True)

        except Exception as e:
            print(f"读取或合并Excel文件 '{filepath}' 失败: {e}。将只写入新数据。")
            combined_df = new_df.reset_index(drop=True) 
    else:
        combined_df = new_df.reset_index(drop=True)

    if 'date' in combined_df.columns and 'hour' in combined_df.columns:
        try:
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
            # Sort by date (as datetime objects for correct chronological order) then by hour
            combined_df.sort_values(
                by=['date', 'hour'], 
                inplace=True, 
                key=lambda col: pd.to_datetime(col, errors='coerce') if col.name == 'date' else col
            )
        except Exception as e:
            print(f"排序时发生错误: {e}. 数据可能未正确排序。")
    
    try:
        combined_df = combined_df[EXPECTED_COLUMNS]
        combined_df.to_excel(filepath, index=False, engine='openpyxl')
        print(f"数据已成功保存到 '{filepath}'")
    except Exception as e:
        print(f"保存到Excel文件 '{filepath}' 失败: {e}")


# --- 主程序 ---
def main_scraper_job(province_name, city_name, station_name): # province_name is kept for consistency but not used in filtering the new API
    print(f"开始爬取任务：省份={province_name}, 城市={city_name}, 站点={station_name} @ {datetime.now()}")
    
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

    session = requests.Session()

    all_data_json = fetch_all_stations_data(session)
    
    if not all_data_json:
        print("未能从API获取到数据，任务终止。")
        return

    # 现在我们直接使用 city_name 和 station_name 来筛选从API获取的数据
    aqi_data_list = parse_and_filter_aqi_data(all_data_json, city_name, station_name)
    
    if not aqi_data_list:
        print(f"未能解析或筛选到目标城市 '{city_name}' 和站点 '{station_name}' 的AQI数据。")
    else:
        print(f"成功解析并筛选到目标站点 '{station_name}' 的 {len(aqi_data_list)} 条AQI数据记录。")

    # 文件名逻辑保持不变
    safe_city_name = re.sub(r'[\\/*?:"<>|]',"", city_name)
    safe_station_name = re.sub(r'[\\/*?:"<>|]',"", station_name) 
    filename = f"{safe_city_name}_{safe_station_name}_AQI_Data.xlsx"
    
    update_excel_data(filename, aqi_data_list)
    print(f"爬取任务完成 @ {datetime.now()}")


if __name__ == "__main__":
    # --- 用户配置 ---
    USER_PROVINCE = "江苏省"  # 省份信息目前主要用于用户识别，API筛选依赖城市和站点名
    USER_CITY = "南京市"      
    USER_STATION = "玄武湖" # (需要与API返回的 positionname 完全一致)
    # --- 用户配置结束 ---
    
    SCRAPE_INTERVAL_SECONDS = 3600 # 1小时
    # SCRAPE_INTERVAL_SECONDS = 20 # For quick testing

    # test_mode 移除了，因为它依赖于旧的HTML片段结构
    # 如果需要测试新的JSON解析，可以保存一个API的JSON响应样本到文件，然后读取并传递给 parse_and_filter_aqi_data

    while True:
        try:
            main_scraper_job(USER_PROVINCE, USER_CITY, USER_STATION)
        except Exception as e:
            print(f"主循环中发生未捕获的严重错误: {e}")
            import traceback
            traceback.print_exc() 
        
        interval_display_unit = "分钟" if SCRAPE_INTERVAL_SECONDS < 3600 else "小时"
        interval_display_value = SCRAPE_INTERVAL_SECONDS / 60 if SCRAPE_INTERVAL_SECONDS < 3600 else SCRAPE_INTERVAL_SECONDS / 3600
        print(f"下一次爬取将在 {interval_display_value:.1f} {interval_display_unit}后进行...")
        time.sleep(SCRAPE_INTERVAL_SECONDS)