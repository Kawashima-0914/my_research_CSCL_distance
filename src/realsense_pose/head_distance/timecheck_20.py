#20分後の終了時刻候補を計算するための小さい補助スクリプト

import pyrealsense2 as rs

def get_bag_start_and_20min_timestamps(bag_file_path: str):
    """
    指定された.bagファイルの開始タイムスタンプと、
    そのタイムスタンプに20分（120万ミリ秒）を足した値を出力する。

    Args:
        bag_file_path (str): .bagファイルのパス

    Returns:
        Tuple[int, int]: 開始タイムスタンプ（ミリ秒）と開始タイムスタンプ+20分（ミリ秒）
    """
    try:
        # RealSenseパイプラインのセットアップ
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(bag_file_path, repeat_playback=False)
        config.enable_stream(rs.stream.depth)

        # パイプライン開始
        print("パイプラインを開始します...")
        pipeline.start(config)
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)

        # 最初のフレームを取得
        print("開始タイムスタンプを取得中...")
        start_timestamp = None

        while True:
            try:
                # フレームを取得
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                depth_frame = frames.get_depth_frame()

                if not depth_frame:
                    continue

                # 最初のタイムスタンプを取得
                start_timestamp = depth_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
                break

            except RuntimeError:
                print("フレーム取得タイムアウト: タイムスタンプを取得できませんでした。")
                break

            # ファイルの終了時にストップ
            if playback.current_status() == rs.playback_status.stopped:
                break

        # パイプライン停止
        pipeline.stop()

        if start_timestamp is None:
            print("開始タイムスタンプが取得できませんでした。")
            return None, None

        # 20分（120万ミリ秒）を加算
        timestamp_20min_later = start_timestamp + 20 * 60 * 1000

        print(f"開始タイムスタンプ: {start_timestamp} ms")
        print(f"20分後のタイムスタンプ: {timestamp_20min_later} ms")

        return start_timestamp, timestamp_20min_later

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None, None


if __name__ == "__main__":
    bag_file_path = "E:/exp-data/20250116_2_CSCL_experiment/20250116_175833.bag"  # ここに.bagファイルのパスを指定
    start, timestamp_20min_later = get_bag_start_and_20min_timestamps(bag_file_path)
    if start is not None and timestamp_20min_later is not None:
        print(f"最終結果: 開始タイムスタンプ: {start} ms, 20分後のタイムスタンプ: {timestamp_20min_later} ms")
    else:
        print("タイムスタンプの取得に失敗しました。")
