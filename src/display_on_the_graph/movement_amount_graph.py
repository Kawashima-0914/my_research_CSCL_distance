from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_one_excel(xlsx_path: Path, output_dir: Path) -> None:
    df = pd.read_excel(xlsx_path)

    required_cols = {"time_sec", "left_distance", "right_distance", "bottom_distance"}
    if not required_cols.issubset(df.columns):
        print(f"スキップ: 必要な列がありません -> {xlsx_path.name}")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df["time_sec"], df["left_distance"], label="Left")
    ax.plot(df["time_sec"], df["right_distance"], label="Right")
    ax.plot(df["time_sec"], df["bottom_distance"], label="Bottom")

    ax.set_title(xlsx_path.stem)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Movement distance (px)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    output_file = output_dir / f"{xlsx_path.stem}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)

    print(f"保存完了: {output_file}")


def main() -> None:
    base_dir = Path(".")
    input_dir = base_dir / "amount_output"
    output_dir = base_dir / "amount_output_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(input_dir.glob("*.xlsx"))

    if not xlsx_files:
        print(f"xlsxが見つかりません: {input_dir.resolve()}")
        return

    print(f"{len(xlsx_files)} 個のxlsxを処理します")

    for xlsx_path in xlsx_files:
        plot_one_excel(xlsx_path, output_dir)

    print("すべて完了しました")


if __name__ == "__main__":
    main()