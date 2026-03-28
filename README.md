# realsense-pose

Realsenseを用いて学習者の頭部間距離を計測するプログラムです。

## うごかしかた

Pythonの仮想環境として[Rye](https://rye-up.com/)を使用しています。
[Ryeのインストールガイド](https://rye-up.com/guide/installation/)を参考に入れてください。

### RealSenseのSDKを入れる
[GitHub](https://github.com/IntelRealSense/librealsense/releases)からIntel RealSense SDKをインストールする。
### macの場合
https://github.com/realsenseai/librealsense/blob/master/doc/installation_osx.md

### 依存の解決

```sh
rye sync
```
### うごかす

```sh
rye run python src/realsense_pose/head_distance/main.py
```


## どういうコードになっているの？

深度カメラからの距離によるマスクを作成し、そのマスクの重心間の距離を学習者の頭部間距離(Distance Between Learners' Heads)としています。


# Measurement_of_inter-head_distance
<img width="362" height="272" alt="image" src="https://github.com/user-attachments/assets/6db509aa-f7e9-470f-a0a5-a4ba5c65f721" />

