#保存済みのフレーム JSON を読んで、フレーム間の時間差が不自然に飛んでいないか確認するチェック用コード

import json
import numpy as np

# JSONファイルをロード
with open('E:/depth_video/20241219_1_CSCL_experiment/5376-7424-5376-7168-1000000frames.json', 'r') as file:
    data = json.load(file)

# タイムスタンプを抽出
timestamps = [frame['timestamp'] for frame in data['data']]

# タイムスタンプ間隔を計算
intervals = np.diff(timestamps)
max_interval = np.max(intervals)
max_interval_index = np.argmax(intervals)

# 異常間隔を検出 (例: 33ms ± 10ms と仮定)
expected_interval = 33
threshold = 10
anomalies = [(i, interval) for i, interval in enumerate(intervals) if abs(interval - expected_interval) > threshold]

# 結果の表示
print(f"異常な間隔: {len(anomalies)} 件")
print(f"最大間隔: {max_interval} ms (インデックス: {max_interval_index})")
for idx, interval in anomalies:
    print(f"フレーム {idx} と {idx+1} の間隔: {interval}ms")
