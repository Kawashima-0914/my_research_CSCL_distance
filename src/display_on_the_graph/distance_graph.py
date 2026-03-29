from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt


def plot_one_json(json_path: Path, output_dir: Path) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"スキップ: 空ファイルです -> {json_path.name}")
        return

    df = pd.DataFrame(data)

    required_cols = {"timestamp", "left_right", "left_bottom", "right_bottom"}
    if not required_cols.issubset(df.columns):
        print(f"スキップ: 必要な列がありません -> {json_path.name}")
        return

    df = df.dropna(subset=["timestamp"]).copy()
    if df.empty:
        print(f"スキップ: timestamp がありません -> {json_path.name}")
        return

    df["elapsed_sec"] = ((df["timestamp"] - df["timestamp"].min()) / 1000.0).astype(int)

    mean_df = (
        df.groupby("elapsed_sec")[["left_right", "left_bottom", "right_bottom"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(mean_df["elapsed_sec"], mean_df["left_right"], label="Left-Right")
    ax.plot(mean_df["elapsed_sec"], mean_df["left_bottom"], label="Left-Bottom")
    ax.plot(mean_df["elapsed_sec"], mean_df["right_bottom"], label="Right-Bottom")

    ax.set_title(f"{json_path.stem} (1-sec average)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (px)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    output_file = output_dir / f"{json_path.stem}_1sec_avg.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"保存完了: {output_file}")


def main() -> None:
    base_dir = Path(".")
    input_dir = base_dir / "output"
    output_dir = base_dir / "output_graphs_distance"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"JSONが見つかりません: {input_dir.resolve()}")
        return

    print(f"{len(json_files)} 個のJSONを処理します")

    for json_path in json_files:
        plot_one_json(json_path, output_dir)

    print("すべて完了しました")


if __name__ == "__main__":
    main()