import datetime
from pathlib import Path
from typing import Optional, TypeVar, Tuple
import os
import numpy as np
import cv2
import pyrealsense2 as rs
import matplotlib.pyplot as plt

T = TypeVar("T")
filename = "/Volumes/SSPA-USC/深度カメラ実験(卒研)および業務オリジナル/exp-data/20241219_1_CSCL_experiment/20241219_145701.bag"


def main():
    rs.log_to_console(rs.log_severity.info)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(filename))
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline_profile = pipeline.start(config)
    if pipeline_profile is None:
        raise RuntimeError("Could not start pipeline (no camera connected?)")

    playback = pipeline_profile.get_device().as_playback()
    playback.set_real_time(False)
    playback.seek(datetime.timedelta(seconds=0))
    i = 0
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        print(depth_frame.get_units())
        print(depth_frame.get_distance(100, 100))
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        print(depth_image[100, 100])
        print(depth_image[100, 100] * depth_frame.get_units())
        # print(i)
        i += 1
        break


if __name__ == "__main__":
    main()
