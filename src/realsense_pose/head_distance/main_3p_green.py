from dataclasses import dataclass
import logging
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import Optional, TypeVar, Tuple
import cv2
import datetime
import json
import matplotlib
import matplotlib.ticker
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas
import pyrealsense2 as rs
import requests
import yaml
from classes import (
    Position,
    Frame,
    DepthFrame,
    Threshold,
    Thresholds,
    Rectangle,
    TimeRange,
    Property,
    CommonConf,
    FilePath,
    FrameSize,
    Mask,
)

T = TypeVar("T")

# 途中経過を表示するかを設定する定数
SHOW_PROGRESS = True

SAVE_DATAFRAME = False

#欠損値をnp.nanに置き換える
def toNdArrayWithNaN(l: list[Optional[T]]) -> np.ndarray:
    return np.array([np.nan if e is None else e for e in l])


#画像を左右に分割する
#始点は画像の左上
#斜めに区切るのはめんどい
#上半分を2分割、下半分はそのまま。
def divide_frame(frame: Frame, left_area_width: int, area_height: int) -> Tuple[Frame, Frame, Frame]:
    #bottomの領域の大きさを他に合わせるか
    #bottom_range = left_area_width * 2
    #bottom_range_xs = int(bottom_range / 4)
    #bottom_range_xe = int((bottom_range / 4) * 3)
    left_depth = frame.depth.image[:area_height, :left_area_width]
    right_depth = frame.depth.image[:area_height, left_area_width:]
    bottom_depth = frame.depth.image[area_height:, :]
    return (
        Frame(DepthFrame(left_depth)),
        Frame(DepthFrame(right_depth)),
        Frame(DepthFrame(bottom_depth))
    )


# 座標原点を揃える
def align_position(position: Position, left_area_size: int) -> Position:
    return Position(position.x + left_area_size, position.y)

# 座標原点をそろえる(bottomのため)
def align_position_y(position: Position, area_height: int) -> Position:
    return Position(position.x, position.y + area_height)


#リスト内の全ての座標を調整
def align_positions(
    positions: list[Optional[Position]], left_area_size: int
) -> list[Optional[Position]]:
    res: list[Optional[Position]] = [None] * len(positions)
    for i, p in enumerate(positions):
        if p is not None:
            res[i] = align_position(p, left_area_size)
    return res


#描画
def annotate_depth_image(
    depth: DepthFrame,
    Positions: Tuple[Optional[Position], Optional[Position], Optional[Position]],
    timestamp: int,
    frame_count: int,
) -> np.ndarray:
    """
    深度画像に重心と線を描画する
    """
    image = depth.image.copy()
    # グレースケールをヒートマップに変換
    image = cv2.applyColorMap(cv2.convertScaleAbs(image, alpha=0.03), cv2.COLORMAP_JET)  # type: ignore
    for p in Positions:
        if p is not None:
            cv2.circle(image, (int(p.x), int(p.y)), 10, (0, 255, 0), -1)  # type: ignore
    # 3点間に線を引く
    if Positions[0] is not None and Positions[1] is not None:
        cv2.line(  # type: ignore
            image,
            (int(Positions[0].x), int(Positions[0].y)),
            (int(Positions[1].x), int(Positions[1].y)),
            (0, 255, 0),
            thickness=2,
        )
    if Positions[1] is not None and Positions[2] is not None:
        cv2.line(
            image,
            (int(Positions[1].x), int(Positions[1].y)),
            (int(Positions[2].x), int(Positions[2].y)),
            (0, 255, 0),
            thickness=2,
        )
    if Positions[2] is not None and Positions[0] is not None:
        cv2.line(
            image,
            (int(Positions[2].x), int(Positions[2].y)),
            (int(Positions[0].x), int(Positions[0].y)),
            (0, 255, 0),
            thickness=2,
        )
    distances: list[Optional[float]] = [None] * 3
    if Positions[0] is not None and Positions[1] is not None:
        distances[0] = abs(Positions[0] - Positions[1])
    if Positions[1] is not None and Positions[2] is not None:
        distances[1] = abs(Positions[1] - Positions[2])
    if Positions[2] is not None and Positions[0] is not None:
        distances[2] = abs(Positions[2] - Positions[0])
    cv2.putText(  # type: ignore
        image,
        f"timestamp: {timestamp}",
        (10, 400),
        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
        1.0,
        (255, 255, 255),
        thickness=2,
    )
    cv2.putText(  # type: ignore
        image,
        f"frame_count: {frame_count}",
        (10, 430),
        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
        1.0,
        (255, 255, 255),
        thickness=2,
    )
    """
    cv2.putText(
        image,
        f"distance 0-1: {'None' if distances[0] is None else f'{distances[0]:.2f}'}",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=2,
    )
    cv2.putText(
        image,
        f"distance 1-2: {'None' if distances[1] is None else f'{distances[1]:.2f}'}",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=2,
    )
    
    cv2.putText(
        image,
        f"distance 2-0: {'None' if distances[2] is None else f'{distances[2]:.2f}'}",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=2,
    ) 
    """
    return image


#設定ファイルyamlを読み込む
def load_config(path: Path) -> Tuple[list[Property], CommonConf]:
    yml = yaml.safe_load(path.read_text(encoding="utf-8"))
    properties: list[Property] = [
        Property(
            project_name=p["project_name"],
            file_path=FilePath(
                project_root=Path(p["file_path"]["project_root"]),
                bag_path=p["file_path"]["bag_path"],
            ),
            time_range=TimeRange(
                start=p["time_range"]["start"],
                end=p["time_range"]["end"],
            ),
            left_area_width=p["left_area_width"],
            area_height=p["area_height"], 
            threshold=Thresholds(
                left=Threshold(
                    int(p["threshold"]["left"]["min"], base=16),
                    int(p["threshold"]["left"]["max"], base=16),
                ),
                right=Threshold(
                    int(p["threshold"]["right"]["min"], base=16),
                    int(p["threshold"]["right"]["max"], base=16),
                ),
                bottom=Threshold(
                    int(p["threshold"]["bottom"]["min"], base=16) if "bottom" in p["threshold"] else 0,
                    int(p["threshold"]["bottom"]["max"], base=16) if "bottom" in p["threshold"] else 0,
                ),
            ),
            target_area=Rectangle(
                lower=Position(
                    p["target_area"]["lower"]["x"], p["target_area"]["lower"]["y"]
                ),
                upper=Position(
                    p["target_area"]["upper"]["x"], p["target_area"]["upper"]["y"]
                ),
            ),
        )
        for p in yml["conf"]
    ]
    common_conf = CommonConf(
        exp_data_root=Path(yml["common"]["exp_data_root"]),
        video_data_root=Path(yml["common"]["video_data_root"]),
    )
    return properties, common_conf

# matplotlibのfigを画像に変換
def fig2img(fig: matplotlib.figure.Figure) -> np.ndarray:
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)  # type: ignore


@dataclass_json
@dataclass(frozen=True)
class DataFrame:
    left: Optional[Position]
    right: Optional[Position]
    bottom: Optional[Position]
    timestamp: Optional[int]

@dataclass_json
@dataclass(frozen=True)
class DataFrames:
    data: list[DataFrame]

    def __len__(self) -> int:
        return len(self.data)
    

#モーメント計算に失敗した場合に、デバッグ用の画像を作成して保存
def create_and_save_failed_moment_image(
    left_mask: Mask,
    right_mask: Mask,
    bottom_mask: Mask,
    start_from_ms: int,
    frame_count: int,
    file_name: Path,
) -> None:
    if all([
        left_mask.moment() is not None, 
        right_mask.moment() is not None,
        bottom_mask.moment() is not None,
        ]):
        return

    # cv2でマスクを2つ繋げて間に余白を入れた画像を作成
    top_row = np.concatenate(  # type: ignore
        (
            left_mask.data,
            np.ones([left_mask.data.shape[0], 10]) * 255,
            right_mask.data,
        ),
        axis=1,
    )

    #bottom_row = cv2.resize(
    #    bottom_mask.data,
    #    (top_row.shape[1], top_row.shape[0]),
    #    interpolation=cv2.INTER_NEAREST,
    #)

    #concat_mask = np.concatenate(
    #   (
    #        top_row,
    #        bottom_row,
    #    ),
    #    axis=0,
    #)

    bottom_row = bottom_mask.data

    concat_mask = np.concatenate(
        (
            top_row,
            np.ones([10, top_row.shape[1]]) * 255,
            bottom_row,
        ),
        axis=0,
    )

    # 画像の上に余白を追加して、そこに文字を入れる
    lined_with_margin = cv2.copyMakeBorder(  # type: ignore
        concat_mask,
        100,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,  # type: ignore
        value=(255, 255, 255),
    )

    mask_img_with_text = cv2.putText(  # type: ignore
        lined_with_margin,
        f"frame: {frame_count}, time: {start_from_ms / 1000:.2f} s",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,  # type: ignore
        1.0,
        (0, 0, 0),
        thickness=2,
    )

    file_name.parent.mkdir(parents=True, exist_ok=True)
    # ファイル名に日本語が含まれているとcv2.imwriteが失敗する
    cv2.imwrite(str(file_name), mask_img_with_text)  # type: ignore
    pass


#フレームを取得し、各フレームのモーメントを計算
def calculate_moments(
    common_conf: CommonConf, property: Property, frame_limit: int
) -> DataFrames:
    config = rs.config()  # type: ignore
    context = rs.context()  # type: ignore
    config.enable_device_from_file(
        str(
            Path(common_conf.exp_data_root)
            / property.file_path.project_root
            / property.file_path.bag_path
        ),
        repeat_playback=False,
    )
    print("calculate_moments が呼び出されました")
    print(f"property.time_range.start: {property.time_range.start}")
    print(f"property.time_range.end: {property.time_range.end}")
    #bagファイルがあるかどうか
    #file_path = Path(common_conf.exp_data_root) / property.file_path.project_root / property.file_path.bag_path

    #if not file_path.exists():
    #    print(f"入力ファイルが存在しません: {file_path}")
    #    return DataFrames(data=[])  # ファイルが存在しない場合、空のデータを返して終了
    #else:
    #    print(f"入力ファイルが確認されました: {file_path}")

    frame_size = FrameSize(640, 480)
    config.enable_stream(
        rs.stream.depth, frame_size.width, frame_size.height, rs.format.z16, 30  # type: ignore
    )
    pipeline = rs.pipeline(context)  # type: ignore
    pipeline_profile = pipeline.start(config)
    if pipeline_profile is None:
        exit(1)

    # こうすると30fpsの動画を30fpsで再生しなくなるので処理速度が速くなる
    # (=30fpsで読むように待たなくなる)
    # https://github.com/IntelRealSense/librealsense/issues/5447#issuecomment-566129013
    profiles = pipeline.get_active_profile()  # type: ignore
    device = profiles.get_device()  # type: ignore

    playback = device.as_playback()  # type: ignore
    playback.set_real_time(False) #問題把握のためにtrueにしてます。

    print("start reading")
    print(f"==={property.project_name}===")
    ok, frames = pipeline.try_wait_for_frames(timeout_ms=100)
    if not ok:
        exit(1)

    # dataframes = [DataFrame(None, None, None)] * frame_limit
    dataframes: list[DataFrame] = []

    head_pos_video_path = (
        common_conf.video_data_root
        / property.project_name
        / f"{property.threshold}-{frame_limit}.mp4"
    )
    head_pos_video_path.parent.mkdir(parents=True, exist_ok=True)
    head_pos_video_handler = cv2.VideoWriter(  # type: ignore
        str(head_pos_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        30,
        (frame_size.width, frame_size.height),
    )

    mask_video_path = (
        common_conf.video_data_root
        / property.project_name
        / f"{property.threshold}-{frame_limit}-mask.mp4"
    )
    mask_video_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dot = np.array(plt.rcParams["figure.figsize"]) * plt.rcParams["figure.dpi"]
    mask_video_handler = cv2.VideoWriter(  # type: ignore
        str(mask_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
        3,
        fig_dot.astype(np.int32),
    )

    mask_plt_fig, mask_plt_axs = plt.subplots(1, 3)
    mask_plt_axs[0].title.set_text("left")
    mask_plt_axs[1].title.set_text("right")
    mask_plt_axs[2].title.set_text("bottom")  

    missed_count = 0
    for i in range(frame_limit):
        ok, frames = pipeline.try_wait_for_frames(timeout_ms=100)
        if not ok:
            print(f"フレーム取得失敗: {i} 番目のフレーム")
            missed_count += 1
            if missed_count > 1:
                print(f"frame length is {i}")
                print("reach to end of file")
                break
            dataframes.append(DataFrame(None, None, None))
            continue
        missed_count = 0
        #print(f"フレーム取得成功: {i} 番目のフレーム")
        try:
            frame = Frame(
                DepthFrame(np.asanyarray(frames.get_depth_frame().get_data()))
            )
            #print(f"フレームデータの一部: {frame.depth.image[:5, :5]}")  # 上位5x5ピクセルを出力
        except ValueError as e:
            print(f"フレームデータの取得に失敗: {e}")
            dataframes.append(DataFrame(None, None, None))
            continue
        timestamp: int = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)  # type: ignore
        #print(f"フレーム {i}: timestamp = {timestamp}")
        if timestamp > property.time_range.end:
            print(f"frame length is {i}")
            print(f"timestamp: {timestamp}")
            print(f"property.time_range.end: {property.time_range.end}")
            print(f"reach to end")
            break

        if i % 1000 == 0:
            print(f"frame progress: {i}")

        left_frame, right_frame, bottom_frame = divide_frame(
            frame, left_area_width=property.left_area_width, area_height=property.area_height,
        )
        left_mask = Mask(
            left_frame.depth, 
            property.threshold.left, 
            target_area=property.target_area
        )
        right_mask = Mask(
            right_frame.depth,
            property.threshold.right,
            target_area=property.target_area,
        )
        bottom_mask = Mask(
            bottom_frame.depth,
            property.threshold.bottom,
            target_area=property.target_area,
        )

        #print(f"left_mask.dataのユニーク値: {np.unique(left_mask.data)}")
        #print(f"right_mask.dataのユニーク値: {np.unique(right_mask.data)}")
        #print(f"bottom_mask.dataのユニーク値: {np.unique(bottom_mask.data)}")
        
        if i % 1000 == 0:
            mask_plt_fig.suptitle(f"frame: {i}")
            mask_plt_axs[0].imshow(left_mask.data)
            mask_plt_axs[1].imshow(right_mask.data)
            mask_plt_axs[2].imshow(bottom_mask.data)

            mask_plt_fig.subplots_adjust(wspace=0.5)

            mask_video_handler.write(cv2.cvtColor(fig2img(mask_plt_fig), cv2.COLOR_RGB2BGR))  # type: ignore

        left_moment = left_mask.moment()
        right_moment = right_mask.moment()
        bottom_moment = bottom_mask.moment()

        #print(f"モーメント (左): {left_moment}")
        #print(f"モーメント (右): {right_moment}")
        #print(f"モーメント (下): {bottom_moment}")

        if left_moment is None or right_moment is None or bottom_mask is None:
            file_name = (
                common_conf.video_data_root
                / property.project_name
                / "images"
                / "calc-moment-failed"
                / f"{property.threshold}-{i}.png"
            )
            create_and_save_failed_moment_image(
                left_mask,
                right_mask,
                bottom_mask,
                start_from_ms=(timestamp - property.time_range.start),
                frame_count=i,
                file_name=file_name,
            )
            print(
                json.dumps(
                    {
                        "msg": "モーメントがどちらかで求められない",
                        "project_name": property.project_name,
                        "left": repr(left_moment),
                        "right": repr(right_moment),
                        "bottom": repr(bottom_moment),
                        "frame": i,
                        "timestamp": (timestamp - property.time_range.start) / 1000,
                    },
                    ensure_ascii=False,
                )
            )
        dataframes.append(DataFrame(left_moment, right_moment, bottom_moment, timestamp))
        head_pos_video_handler.write(
            annotate_depth_image(
                frame.depth,
                (
                    left_moment,
                    align_position(
                        right_moment, left_area_size=property.left_area_width
                    )
                    if right_moment is not None
                    else None,
                    align_position_y(
                        bottom_moment, area_height=property.area_height
                    )
                    if bottom_moment is not None
                    else None,
                ),
                timestamp=(timestamp - property.time_range.start) / 1000,  # type: ignore
                frame_count=i,
            )
        )
    pipeline.stop()
    head_pos_video_handler.release()
    mask_video_handler.release()
    return DataFrames(data=dataframes)


#コマンドライン引数からターゲットプロジェクトを取得します。
def get_targets() -> list[str]:
    import sys

    args = sys.argv[1:]
    if "all" in args:
        return ["all"]
    return args


#スクリプトの使い方を説明する文字列を返す
def usage():
    return """
    全てのデータを集計する
    python main_3p_green.py all

    特定の実験データを集計する
    python main_3p_green.py 20241219_1_CSCL_experiment 20241219_1_CSCL_experiment
"""


#計算したモーメントデータをjson形式で保存
def save_positions(positions: DataFrames, filename: Path) -> None:
    #print(f"キャッシュファイルを保存中: {filename}")
    #print(f"保存するデータ: {positions}")
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        positions_json = positions.to_json(ensure_ascii=False)  # type: ignore
        #print(f"JSONデータ: {positions_json[:500]}")
        f.write(positions_json)

#保存されたjsonファイルを読み込み、モーメントデータとして返す
def load_positions(filename: Path) -> DataFrames:
    with open(filename, "r") as f:
        positions_json = f.read()
        return DataFrames.from_json(positions_json)  # type: ignore
    

#3領域のモーメント間の距離を計算する
def calc_distances(positions: DataFrames, left_area_size: int, area_height: int) -> list[dict[str, Optional[float]]]:
    #3領域のモーメント間の距離を計算
    #ex) 左上と下→left_bottom
    distances: list[dict[str, Optional[float]]] = [None] * len(positions)

    for i, position in enumerate(positions.data):
        left = position.left
        right = position.right
        bottom = position.bottom

        #初期化：距離が計算できない場合はNone
        distances[i] = {
            "left_right": None,
            "left_bottom": None,
            "right_bottom": None,
        }
        
        # 左上と右上の距離
        if left is not None and right is not None:
            aligned_right = align_position(right, left_area_size)
            distances[i]["left_right"] = abs(left - aligned_right)

        #左上と下の距離
        if left is not None and bottom is not None:
            aligned_bottom = align_position_y(bottom, area_height)
            distances[i]["left_bottom"] = abs(left - aligned_bottom)

        #右上と下の距離
        if right is not None and bottom is not None:
            aligned_right = align_position(right, left_area_size)
            aligned_bottom = align_position_y(bottom, area_height)
            distances[i]["right_bottom"] = abs(aligned_right - aligned_bottom)

        
    return distances


#距離とタイムスタンプを保管するよう(calc_distancesとは念のために別にしてます)
def calc_distances_with_timestamps(positions: DataFrames, left_area_size: int, area_height: int) -> list[dict]:
    distances_with_timestamps = []

    for position in positions.data:
        left = position.left
        right = position.right
        bottom = position.bottom

        # 初期化（距離が計算できない場合はNone）
        distances = {
            "timestamp": position.timestamp,  # タイムスタンプを追加
            "left_right": None,
            "left_bottom": None,
            "right_bottom": None,
        }

        # 左上と右上の距離
        if left is not None and right is not None:
            aligned_right = align_position(right, left_area_size)
            distances["left_right"] = abs(left - aligned_right)

        # 左上と下の距離
        if left is not None and bottom is not None:
            aligned_bottom = align_position_y(bottom, area_height)
            distances["left_bottom"] = abs(left - aligned_bottom)

        # 右上と下の距離
        if right is not None and bottom is not None:
            aligned_right = align_position(right, left_area_size)
            aligned_bottom = align_position_y(bottom, area_height)
            distances["right_bottom"] = abs(aligned_right - aligned_bottom)

        distances_with_timestamps.append(distances)

    return distances_with_timestamps

def save_distances_to_json(distances_with_timestamps: list[dict], filename: Path) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)  # ディレクトリを作成
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(distances_with_timestamps, f, ensure_ascii=False, indent=4)



#スクリプト全体のエントリーポイント
# 1.設定を読み込む
# 2.モーメントを計算し、可視化
# 3.データを保存
# 4.処理終了を通知
def main():
    logging.basicConfig(level=logging.INFO)
    results: list[DataFrames] = []
    distance_sequences: list[list[dict[str, Optional[float]]]] = []
    import os
    print(f"Current working directory: {os.getcwd()}")
    properties, common_conf = load_config(Path(".") / "config.yaml")
    rs.log_to_console(rs.log_severity.info)  # type: ignore

    targets = get_targets()
    filtered_properties = (
        list(filter(lambda p: p.project_name in targets, properties))
        if not "all" in targets
        else properties
    )

    #print(f"filtered_propertiesの内容: {filtered_properties}")
    if len(filtered_properties) == 0:
        print(usage())
        exit(1)

    logging.info(
        f"target is {list(map(lambda p: p.project_name, filtered_properties))}"
    )

    for property in filtered_properties:
        now = datetime.datetime.now()
        logging.info(f'start_at {now.strftime("%Y/%m/%d %H:%M:%S")}')
        frame_limit = 1000000

        positions_json_path = (
            common_conf.video_data_root
            / property.project_name
            / f"{property.threshold}-{frame_limit}frames.json"
        )

        use_cache = True
        is_cache_exist = positions_json_path.exists() and use_cache

        #print(f"キャッシュファイルのパス: {positions_json_path}")
        #print(f"キャッシュファイル存在: {positions_json_path.exists()}")

        def get_positions_from_cache_or_calc() -> DataFrames:
            print("get_posiは呼び出された")
            if is_cache_exist:
                logging.info(f"load from cache {positions_json_path}")
                return load_positions(positions_json_path)

            print("キャッシュが存在しないため、calculate_momentsを実行")
            return calculate_moments(common_conf, property, frame_limit)

        try:
            positions = get_positions_from_cache_or_calc()
            #print(f"positions.dataの長さ: {len(positions.data)}")
            #print(f"positions.dataの内容: {positions.data[:10]}")  # 最初の10個を確認
        except Exception as e:
            print(e)
            print("failed to calc")
            continue

        print(
            "elapsed_time", f"{(datetime.datetime.now() - now).total_seconds():.2f} s"
        )
        results.append(positions)

        if not is_cache_exist:
            logging.info(f"save to cache {positions_json_path}")
            save_positions(positions, positions_json_path)


        distance_sequence = calc_distances(positions, property.left_area_width, property.area_height)
        #print(f"distance_sequenceの長さ: {len(distance_sequence)}")
        #print(f"distance_sequenceの内容: {distance_sequence[:10]}")  # 最初の10個を確認
        distance_sequences.append(distance_sequence)

        # 距離計算（タイムスタンプ付き）
        distance_sequence_with_timestamps = calc_distances_with_timestamps(
            positions, property.left_area_width, property.area_height
        )

        # JSONファイルとして保存
        json_output_path = pathlib.Path(".") / "output" / f"{property.project_name}_distances.json"
        save_distances_to_json(distance_sequence_with_timestamps, json_output_path)

        logging.info(f"Distances saved to {json_output_path}")

        image_path = (
            pathlib.Path(".")
            / "images"
            / property.project_name
            / f"between-length-{property.threshold}-{len(positions)}frames-green.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        timestamps_ms = (
            toNdArrayWithNaN(list(map(lambda t: t.timestamp, positions.data)))
            - property.time_range.start
        )

        time_stamps_at_0_idx = np.where(timestamps_ms < 0)[0].shape[0]

        # デバッグコード
        #print(f"デバッグ: timestamps_ms.shape = {(timestamps_ms / 1000)[time_stamps_at_0_idx:].shape}")
        #print(f"デバッグ: distance_sequence の長さ = {len(distance_sequence)}")
        #print(f"デバッグ: left_right データ例 = {[d['left_right'] for d in distance_sequence][:5]}")

        #データ長をそろえるための処理
        min_length = min(
            len((timestamps_ms / 1000)[time_stamps_at_0_idx:]),
            len([d["left_right"] for d in distance_sequence][time_stamps_at_0_idx:]),
            len([d["left_bottom"] for d in distance_sequence][time_stamps_at_0_idx:]),
            len([d["right_bottom"] for d in distance_sequence][time_stamps_at_0_idx:])
        )

        rcParams_val = {
            "figure.figsize": (20, 10),
            "figure.dpi": 200,
            "font.size": 32,
            "lines.color": "g",
        }
        with matplotlib.rc_context(rcParams_val):
            fig, ax = plt.subplots()
            fig.suptitle("Distance Between Learners’ Heads")
            ax.plot(
                (timestamps_ms / 1000)[time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],  # type: ignore
                [d["left_right"] for d in distance_sequence][time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],
                label="Left-Right",
                # label=result.property.project_name,
                color="g",
            )
            ax.plot(
                (timestamps_ms / 1000)[time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],  # type: ignore
                [d["left_bottom"] for d in distance_sequence][time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],
                label="Left-Bottom",
                # label=result.property.project_name,
                color="b",
            )
            ax.plot(
                (timestamps_ms / 1000)[time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],  # type: ignore
                [d["right_bottom"] for d in distance_sequence][time_stamps_at_0_idx:time_stamps_at_0_idx + min_length],
                label="Right-Bottom",
                # label=result.property.project_name,
                color="r",
            )
            # 横軸に3桁区切りのカンマを入れる
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
            )

            # ax.plot(
            #     ((toNdArrayWithNaN(result.timestamp) - result.property.time_range.start) / 1000)[time_stamps_at_0_idx:],  # type: ignore
            #     moving_average[time_stamps_at_0_idx:],
            #     label=f"moving average ({moving_window_size})",
            #     color="b",
            # )
            ax.legend()
            ax.set_xlabel("time (s)")
            ax.set_ylabel("distance (pixel)")
            fig.savefig(image_path)

    webhook_endpoint = "https://discord.com/api/webhooks/1119943092292829194/s6a-cVHQpLiHzXpkwELG-3xSvcPUBbOB6RmiDPmlDVt1CxfRFxRc1_3WxtJ46Zgs8ehy"
    requests.post(
        webhook_endpoint,
        data=json.dumps(
            {
                "content": "\n".join(
                    [
                        f"データ集計が終了しました",
                        f"集計対象 {' '.join(targets)}",
                    ]
                ),
            }
        ),
        headers={"Content-Type": "application/json"},
    )
    print("sended_at", datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

    # distance_sequences を np.ndarray に変換する
    # seaborn.violinplot(data=pandas_dataframe, ax=ax)
    # ax.set_xlabel("group")
    # ax.set_ylabel("distance (pixel)")
    # pandas_dataframe.boxplot(ax=ax, showfliers=False)
    # pandas_dataframe = pd.DataFrame(
    #     {
    #         p.project_name: pd.Series(toNdArrayWithNaN(d))
    #         for p, d in zip(filtered_properties, distance_sequences)
    #     }
    # )
    pandas_df = pandas.DataFrame(
        {
            p.project_name: pandas.Series(toNdArrayWithNaN(d))
            for p, d in zip(filtered_properties, distance_sequences)
        }
    )
    if SAVE_DATAFRAME:
        pandas_df.to_csv(
            Path(".") / "images" / "distance-between-learners-heads.csv", index=False
        )
        # pandas_dataframe.to_csv(
        #     Path(".") / "images" / "distance-between-learners-heads.csv", index=False
        # )
    # ax.boxplot(
    #     pandas_dataframe,
    #     labels=[p.project_name for p in filtered_properties],
    #     showfliers=False,
    # )
    # fig.savefig(
    #     pathlib.Path(".") / "images" / f"boxplot-distance-between-learners-heads.png"
    # )


"""
全てのデータを集計する
python main.py all

特定の実験データを集計する
python main.py 20231031_1_CSCL_experiment 20231102_1_CSCL_experiment
"""
if __name__ == "__main__":
    main()