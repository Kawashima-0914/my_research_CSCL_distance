from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Self
from dataclasses_json import dataclass_json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2


@dataclass_json
@dataclass
class Position:
    x: float
    y: float

    def __add__(self, other) -> Self:
        return Position(self.x + other.x, self.y + other.y)  # type: ignore

    def __sub__(self, other) -> Self:
        return Position(self.x - other.x, self.y - other.y)  # type: ignore

    def __abs__(self) -> float:
        return np.sqrt(self.x**2 + self.y**2)

    def __lt__(self, other) -> bool:
        return abs(self) < abs(other)

    def __gt__(self, other) -> bool:
        return abs(self) > abs(other)


@dataclass(frozen=True)
class DepthFrame:
    image: np.ndarray

    def __post_init__(self):
        if self.image.all() == None:
            raise ValueError("image is empty")

    def smooth(self, ksize: int) -> Self:
        return DepthFrame(
            cv2.GaussianBlur(self.image, (ksize, ksize), 0),  # type: ignore
        )

    def show(self, plt=plt, vmin: int = 1000, vmax: int = 6000) -> None:
        sns.heatmap(self.image, vmin=vmin, vmax=vmax, cmap="jet")
        plt.show()

    def heatmap(self, vmin: int = 1000, vmax: int = 6000):
        return sns.heatmap(self.image, vmin=vmin, vmax=vmax, cmap="jet")


@dataclass(frozen=True)
class Frame:
    depth: DepthFrame
    # color: ColorFrame


@dataclass
class Threshold:
    min: float
    max: float

    def __post_init__(self):
        if self.min > self.max:
            raise ValueError("min is greater than max")


@dataclass
class FrameSize:
    width: int
    height: int


@dataclass(frozen=True)
class Thresholds:
    left: Threshold
    right: Threshold
    bottom: Threshold

    def __str__(self):
        return f"{self.left.min}-{self.left.max}-{self.right.min}-{self.right.max}"

@dataclass(frozen=True)
class Rectangle:
    lower: Position
    upper: Position

    def __post_init__(self):
        if self.lower > self.upper:
            raise ValueError("lower is greater than upper")

    def __contains__(self, position: Position) -> bool:
        return (
            self.lower.x <= position.x <= self.upper.x
            and self.lower.y <= position.y <= self.upper.y
        )


@dataclass(frozen=True)
class TimeRange:
    start: int
    end: int

    def __post_init__(self):
        if self.start > self.end:
            raise ValueError("start is greater than end")


@dataclass(frozen=True)
class DistDir:
    video_dir: Path
    image_dir: Path


@dataclass(frozen=True)
class FilePath:
    project_root: Path
    bag_path: str


@dataclass(frozen=True)
class Property:
    project_name: str
    file_path: FilePath
    left_area_width: int
    area_height: int
    threshold: Thresholds
    time_range: TimeRange
    target_area: Rectangle


@dataclass(frozen=True)
class CommonConf:
    exp_data_root: Path
    video_data_root: Path


class Mask:
    data: np.ndarray
    threshold: Threshold

    def __init__(
        self,
        frame: DepthFrame,
        threshold: Threshold,
        target_area: Rectangle = Rectangle(
            lower=Position(0, 0), upper=Position(640, 480)
        ),
    ) -> None:
        target_mask = np.zeros_like(frame.image)
        target_mask[
            int(target_area.lower.y) : int(target_area.upper.y),
            int(target_area.lower.x) : int(target_area.upper.x),
        ] = 1
        # target_maskをCV_8Uにする
        target_mask = target_mask.astype(np.uint8)
        self.data = cv2.inRange(cv2.bitwise_and(frame.image, frame.image, mask=target_mask), threshold.min, threshold.max)  # type: ignore
        self.threshold = threshold

    def moment(self) -> Optional[Position]:
        moment = cv2.moments(self.data)  # type: ignore
        if moment["m00"] == 0:
            return None
        return Position(moment["m10"] / moment["m00"], moment["m01"] / moment["m00"])
