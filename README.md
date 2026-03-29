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
rye run python src/realsense_pose/head_distance/main_3p_green.py
```


## どういうコードになっているの？

深度カメラからの距離によるマスクを作成し、そのマスクの重心間の距離を学習者の頭部間距離(Distance Between Learners' Heads)としています。


# Measurement_of_inter-head_distance
3人のディスカッション中の頭部動作を計測するプロジェクトです。
下記のような議論状況となっております。
上記コードを動かすと、下記のようなマスク画像と、頭部間の距離を可視化した動画、2者間の距離および、各参加者の頭部座標を示しています。頭部座標は、頭部として判定される領域の重心座標となっています。

<img width="368" height="275" alt="image" src="https://github.com/user-attachments/assets/c470374f-a8da-4730-a3c4-02d9571d7472" />
<img width="343" height="255" alt="image" src="https://github.com/user-attachments/assets/a47d52a3-6b4c-40b1-b485-35bc1c054370" />
<img width="362" height="272" alt="image" src="https://github.com/user-attachments/assets/6db509aa-f7e9-470f-a0a5-a4ba5c65f721" />


# Measurement_of_head_movement_amount
src/calculate_movement_amount/amount_move_distance_new.pyは、depth_videoに格納されている各参加者の頭部座標から、1秒間あたりの頭部動作量を算出し、amount_outputに出力する

# graph
頭部間距離および頭部動作量をグラフへプロットするソースコードを、src/display_on_the_graphに追加しています。
以下はその出力されたグラフの例です。

<img width="1257" height="623" alt="image" src="https://github.com/user-attachments/assets/58c221f0-37c0-4a85-9219-8336304af029" />

<img width="1093" height="541" alt="image" src="https://github.com/user-attachments/assets/60f6b511-3e2a-4ac0-a779-bed1de4ba834" />

