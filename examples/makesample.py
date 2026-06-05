"""
DonutCode 統合テスト＆デバッグスクリプト

【概要】
このスクリプトは、DonutCodeライブラリの動作確認を行うための総合テストツールです。
以下の2つの処理を連続して実行します。

1. [エンコード]: 指定された設定（サイズ・穴の位置・エラー訂正・テキスト）から
   DonutCodeを生成し、画像として保存します。
2. [デコード＆デバッグ]: 保存された画像を読み込み、OpenCVを用いて段階的に解析します。
   解析の各ステップ（領域検出、正面補正、白黒サンプリング）の画像を保存し、
   最終的な文字列の復元（デコード）を行います。

【出力先】
すべての生成画像およびデバッグ画像は `sample-result/` フォルダに出力されます。
"""

import os
import cv2
import numpy as np
import donutcode

# ==========================================
# テスト用設定 
# ==========================================
#無駄がないパターン
GRID_SIZE = 27
HOLE_RECT = (7, 7, 13, 13)
ECC_BYTES = 24

# 小数点4桁（11メートル精度）にしています。
OUTPUT_DIR = "sample-donuts"

# 出力先フォルダの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# デバッグ用補助関数
# ==========================================
def _is_reserved_pattern(col, row, grid_size):
    # 3隅のファインダーパターン (7x7)
    if 0 <= col < 7 and 0 <= row < 7: return True
    if grid_size - 7 <= col < grid_size and 0 <= row < 7: return True
    if 0 <= col < 7 and grid_size - 7 <= row < grid_size: return True
    # 右下のアライメントアンカー (3x3)
    if grid_size - 3 <= col < grid_size and grid_size - 3 <= row < grid_size: return True
    return False

def _is_hole(col, row, hole_rect):
    hx, hy, hw, hh = hole_rect
    return hx <= col < hx + hw and hy <= row < hy + hh

def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

import random

def generate_random_location():
    # 緯度 (Latitude): -90度から90度
    lat = random.uniform(-90.0, 90.0)
    # 経度 (Longitude): -180度から180度
    lon = random.uniform(-180.0, 180.0)
    
    # 小数点以下4桁で文字列フォーマット
    return f"loc:{lat:.4f},{lon:.4f}"

# ==========================================
# メイン処理
# ==========================================
def main():
    print("===== DonutCode サンプル作成 =====")
    
    for n in range(30):
        # ---------------------------------------------------------
        # 1. 画像の生成 (エンコード)
        # ---------------------------------------------------------
        TEST_MESSAGE = generate_random_location()
        OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, f"test_fresh_donut_27_{n}.png")
        print(f"\n[1] エンコードを開始します (メッセージ: '{TEST_MESSAGE}')")
        try:
            encoder = donutcode.Encoder(
                grid_size=GRID_SIZE, 
                hole_rect=HOLE_RECT, 
                ecc_bytes=ECC_BYTES
            )
            matrix = encoder.encode(TEST_MESSAGE)
            # 画像を sample-result 内に出力
            encoder.save_image(matrix, OUTPUT_IMAGE, scale=15, hole_color="#ffebee")
            print(f" -> 新しい画像 '{OUTPUT_IMAGE}' の生成に成功しました！")
            
        except Exception as e:
            print(f"❌ エンコード中にエラーが発生しました: {e}")
            return



if __name__ == "__main__":
    main()