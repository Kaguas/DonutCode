"""
DonutCode サンプル一括生成スクリプト

【概要】
このスクリプトは、ランダムな緯度・経度の座標情報を持つDonutCode画像を
連続して複数枚生成し、指定したフォルダに出力します。
"""

import os
import random
from donutcode import Encoder

# ==========================================
# 設定 
# ==========================================
CONFIG_TYPE = "D-27-13"
OUTPUT_DIR = "sample-donuts"
NUM_SAMPLES = 30 # 生成する画像の枚数

# 出力先フォルダの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# ランダム座標生成関数
# ==========================================
def generate_random_location():
    # 緯度 (Latitude): -90度から90度
    lat = random.uniform(-90.0, 90.0)
    # 経度 (Longitude): -180度から180度
    lon = random.uniform(-180.0, 180.0)
    
    # 小数点以下4桁で文字列フォーマット
    return f"{lat:.4f},{lon:.4f}"

# ==========================================
# メイン処理
# ==========================================
def main():
    print(f"===== DonutCode サンプル一括作成開始 ({NUM_SAMPLES}件) =====")
    
    try:
        # 最新版APIに合わせてコンフィグを指定してエンコーダを初期化
        # ※初期化はループ外で1度だけ行うと処理が高速になります
        encoder = Encoder(config_type=CONFIG_TYPE)
    except Exception as e:
        print(f"❌ エンコーダの初期化に失敗しました: {e}")
        return

    for n in range(NUM_SAMPLES):
        test_message = generate_random_location()
        # ファイル名がソートしやすいようにゼロ埋め（例: _00.png, _01.png...）
        output_image = os.path.join(OUTPUT_DIR, f"test_fresh_donut_{n:02d}.png")
        
        print(f"[{n+1:02d}/{NUM_SAMPLES}] エンコード: '{test_message}'")
        try:
            # エンコード実行
            matrix = encoder.encode(test_message)
            # 画像の保存
            encoder.save_image(matrix, output_image, scale=15, hole_color="#ffebee")
            print(f" -> 生成成功: {output_image}")
            
        except Exception as e:
            print(f"❌ 画像の生成中にエラーが発生しました: {e}")

    print("\n===== 全サンプルの生成が完了しました =====")

if __name__ == "__main__":
    main()