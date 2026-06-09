# DonutCode (v1.0.0)

DonutCode（ドーナツコード）は、中央に指定したサイズの「穴（中抜き領域）」を配置できる独自の2次元コードを生成・解析するためのPythonライブラリです。

「QRコード」とは異なる完全な独自規格であり、中央の巨大な空白スペースにデジタルサイネージ、動的情報、物理的なポール、または独自のロゴなどを自由に配置できることを最大の強みとしています。

## 主な特徴

巨大で自由な「穴」: コードの中央を任意のサイズでくり抜くことができます。

強力なエラー訂正: リード・ソロモン符号によるエラー訂正機能を搭載。全体の数割が欠損・汚損してもデータを復元できます。

遠距離・カメラ読み取りへの最適化: 四隅のうち3箇所にファインダーパターン、右下に3x3のアライメントアンカーを配置。OpenCVの透視変換（パースペクティブ補正）により、斜めから撮影した歪んだ画像からでも正確にデータを抽出します。

柔軟なサイズ設計: グリッドサイズ、穴のサイズ、エラー訂正の強度を要件（コンパクトさ優先 or 遠距離・高耐久優先）に合わせて1バイト単位でチューニング可能です。


## インストール方法

リポジトリをダウンロード後、コマンドラインから以下のスクリプトを実行してインストールしてください。

[Windowsの場合]
install.bat をダブルクリックするか、コマンドプロンプトで実行します。
``` (PowerShell)
install.bat
```

[Mac / Linuxの場合]
``` bash
chmod +x install.sh;
./install.sh
```
※ 内部的には pip install -e . を実行し、エディタブルモードでインストールしています。

## 必要なライブラリ (requirements.txt)

opencv-python >= 4.0.0
numpy >= 1.20.0
Pillow >= 9.0.0

## パッケージ構成
```
DonutCode/
|-- pyproject.toml
|-- README.md
|-- requirements.txt
|-- install.bat
|-- install.sh
|-- src/
|   -- donutcode/ 
|       |-- __init__.py      (モジュールの公開) 
|       |-- encoder.py       (生成ロジック) 
|       |-- decoder.py       (解析・解読ロジック) 
|       |-- reedsolomon.py   (エラー訂正ロジック)
|       |-- vision.py   (Decoder用の検出用画像処理)
-- examples/ 
   |-- sample.py          (画像生成とデバッグ解析の統合テスト) 
   |-- scanner.py         (Webカメラを使ったリアルタイムスキャナー)
```

## 基本的な使い方

### 1. エンコード (画像の生成)
Pythonスクリプトから donutcode をインポートし、設定を指定して画像を生成します。
``` 
from donutcode import Encoder
``` 
#### [設定例: 27x27のグリッド、中央に13x13の穴]
configであらかじめ定義したバージョンとサイズを指定することでエンコーダを初期化
```
# コンフィグを指定してエンコーダを初期化
encoder = Encoder("D-27-13")

# マッピング画像の生成
# DEBUG_MAPPING_IMAGE = "00_mapping_debug.png"
# encoder.save_mapping_debug_image(DEBUG_MAPPING_IMAGE, scale=20, padding=20)
# print(f" -> [DEBUG] マッピング確認画像を生成しました: {DEBUG_MAPPING_IMAGE}")

# 本番のエンコード画像生成
matrix = encoder.encode(TEST_MESSAGE)
encoder.save_image(matrix, OUTPUT_IMAGE, scale=15, hole_color="#ffebee")
```

### 2. デコード (画像ファイルからの読み取り)
生成時と同じグリッドサイズと穴のサイズを指定してデコーダーを初期化します。
```
from donutcode import Decoder
```
#### [設定例: 27x27のグリッド、中央に13x13の穴]
configであらかじめ定義したバージョンとサイズを指定することでエンコーダを初期化
```
# コンフィグを指定してデコーダを初期化
decoder = Decoder(config_type=CONFIG_TYPE)
        
# 内部で VisionProcessor による補正からデータ復元まで一気通貫で行われます
result = decoder.decode_image(OUTPUT_IMAGE)
```
### 3. 旧リアルタイム・スキャナー
カメラ映像からリアルタイムにDonutCodeを検知し、空間にデコード結果をAR表示するサンプルスクリプトが付属しています。
```
# サンプルとして残していますが最新版に対応していません
python examples/scanner.py
```
## ユースケースと推奨設定（ベストプラクティス）
生成とデコードの一貫テストサンプル
```
python3 examples/sample.py
```

デコードテスト
```
python decoder_test.py my_photo.jpg --config D-27-13
```


## 更新履歴
v1.0.0 モックアップとして公開<br>
v1.0.1 ファインダパタン7*7の周り1マスは空白として開けるように改良しました。<br>
v1.0.2 Donutcodeの周りにパディングを追加しました<br>
V1.1.0 タイミングパタン等を追加しリリース　（27_13のみ）<br>
