# AnimeGANv2-Face-Overlay-Demo
<img src="https://user-images.githubusercontent.com/37477845/141478903-c06a1e89-54c1-4982-b044-582ef9711d98.gif" width="75%"><br>
[PyTorch Implementation of AnimeGANv2](https://github.com/bryandlee/animegan2-pytorch)を用いて、生成した顔画像を元の画像に上書きするデモです。<br> 

# Requirement
* mediapipe 0.8.9 or later
* OpenCV 4.5.3.56 or later
* onnxruntime-gpu 1.9.0 or later <br>※onnxruntimeでも動作しますが、推論時間がかかるのでGPUをお勧めします

### 処理速度参考値
GeForce GTX 1050 Ti：約3.3fps<br>
GeForce RTX 3060：約9fps

# Demo
デモの実行方法は以下です。
```bash
python main.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --fd_model_selection<br>
顔検出モデル選択(0：2m以内の検出に最適なモデル、1：5m以内の検出に最適なモデル)<br>
デフォルト：model/face_paint_512_v2_0.onnx
* --min_detection_confidence<br>
顔検出信頼値の閾値<br>
デフォルト：0.5
* --animegan_model<br>
AnimeGANv2のモデル格納パス<br>
デフォルト：model/face_paint_512_v2_0.onnx
* --animegan_input_size<br>
AnimeGANv2のモデルの入力サイズ<br>
デフォルト：512
* --ss_model_selection<br>
モデル種類指定<br>
0：Generalモデル(256x256x1 出力)<br>
1：Landscapeモデル(144x256x1 出力)<br>
デフォルト：0
* --ss_score_th<br>
スコア閾値(閾値以上：人間、閾値未満：背景)<br>
デフォルト：0.1
* --debug<br>
デバッグウィンドウを表示するか否か<br>
デフォルト：指定なし
* --debug_subwindow_ratio<br>
デバッグウィンドウの拡大率<br>
デフォルト：0.5

※デバッグ表示有効時は以下のようなウィンドウを表示<br>
<img src="https://user-images.githubusercontent.com/37477845/141482433-bed41338-2af1-41b9-8af7-fcbad15f66f1.gif" width="75%"><br>

# Reference
* [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
AnimeGANv2-Face-Overlay-Demo is under [MIT License](LICENSE).
