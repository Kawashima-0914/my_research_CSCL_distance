import pandas as pd
import json
import numpy as np
import os

# JSONファイルを読み込む
print(os.getcwd())
file_path = os.path.join('depth_video', '20241219_1_CSCL_experiment', '5376-7424-5376-7168-1000000frames.json')
with open(file_path, 'r') as file:
    data = json.load(file)['data']

# DataFrameに変換
df = pd.DataFrame(data)

# タイムスタンプと座標展開
df['timestamp'] = df['timestamp'] - df['timestamp'].min()  # 開始時点を0に調整
df['time_sec'] = (df['timestamp'] / 1000).astype(int)  # 秒単位に変換
df = pd.concat([df.drop(['left', 'right', 'bottom'], axis=1), 
                df['left'].apply(pd.Series).add_prefix('left_'), 
                df['right'].apply(pd.Series).add_prefix('right_'), 
                df['bottom'].apply(pd.Series).add_prefix('bottom_')], axis=1)

# 1秒間ごとの平均座標を計算
mean_per_second = df.groupby('time_sec').mean().reset_index()

# フレーム間の移動距離（ユークリッド距離）を計算
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

mean_per_second['left_distance'] = calculate_distance(
    mean_per_second['left_x'].shift(), mean_per_second['left_y'].shift(),
    mean_per_second['left_x'], mean_per_second['left_y']
)
mean_per_second['right_distance'] = calculate_distance(
    mean_per_second['right_x'].shift(), mean_per_second['right_y'].shift(),
    mean_per_second['right_x'], mean_per_second['right_y']
)
mean_per_second['bottom_distance'] = calculate_distance(
    mean_per_second['bottom_x'].shift(), mean_per_second['bottom_y'].shift(),
    mean_per_second['bottom_x'], mean_per_second['bottom_y']
)

mean_per_second = mean_per_second.fillna(0)  # 最初のフレームのNaNを0に置き換え

# 結果を整理
result = mean_per_second[['time_sec', 'left_distance', 'right_distance', 'bottom_distance']]

# ファイル出力
output_file = 'amount_output/coordinates_avg_distance_per_second.xlsx'
result.to_excel(output_file, index=False, engine='openpyxl')

print(f"1秒間の平均座標間の移動距離がExcelファイルに出力されました: {output_file}")
