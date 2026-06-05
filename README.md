# DonutCode (v1.0.2)

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

install.bat

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
-- examples/ 
   |-- sample.py          (画像生成とデバッグ解析の統合テスト) 
   |-- scanner.py         (Webカメラを使ったリアルタイムスキャナー)

## 基本的な使い方

### 1. エンコード (画像の生成)
Pythonスクリプトから donutcode をインポートし、設定を指定して画像を生成します。

from donutcode import Encoder

[設定例: 25x25のグリッド、中央に11x11の穴、エラー訂正22バイト]
encoder = Encoder(grid_size=25, hole_rect=(7, 7, 11, 11), ecc_bytes=22)

matrix = encoder.encode("loc:34.2335,133.6387")
encoder.save_image(matrix, "my_donut.png", scale=15, hole_color="#ffebee")

### 2. デコード (画像ファイルからの読み取り)
生成時と同じグリッドサイズと穴のサイズを指定してデコーダーを初期化します。

from donutcode import Decoder

decoder = Decoder(grid_size=25, hole_rect=(7, 7, 11, 11))
results = decoder.decode_image("my_donut.png")

if results:
print("デコード成功:", results[0])

### 3. リアルタイム・スキャナー
カメラ映像からリアルタイムにDonutCodeを検知し、空間にデコード結果をAR表示するサンプルスクリプトが付属しています。

python examples/scanner.py

## ユースケースと推奨設定（ベストプラクティス）

ドローンや定点カメラからの「遠距離スキャン」を想定する場合、コード全体のマス目を極力減らし、1マスあたりの物理的な大きさを最大化することが重要です。以下の2パターンの設定は、1ビットの無駄も出ない数学的な最適解です。

[パターンA: 穴サイズ優先・標準耐久モデル]
全体の約20%を穴にしつつ、実用的なエラー訂正を確保します。

GRID_SIZE = 25
HOLE_RECT = (7, 7, 11, 11)
ECC_BYTES = 22
(データ容量: 最大22バイト)

[パターンB: 超巨大穴・高耐久モデル]
全体の約23%を穴にしつつ、全体の約30%の欠損に耐える高信頼モデルです。

GRID_SIZE = 27
HOLE_RECT = (7, 7, 13, 13)
ECC_BYTES = 29
(データ容量: 最大22バイト)



## 更新履歴
v1.0.0 モックアップとして公開<br>
v1.0.1 ファインダパタン7*7の周り1マスは空白として開けるように改良しました。<br>
v1.0.2 Donutcodeの周りにパディングを追加しました<br>