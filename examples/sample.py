"""
DonutCode 統合テスト＆デバッグスクリプト

【概要】
このスクリプトは、DonutCodeライブラリの動作確認を行うための総合テストツールです。
以下の2つの処理を連続して実行します。

1. [エンコード]: 指定された設定からDonutCodeを生成し、画像として保存します。
   ※合わせて、デバッグ用の「マッピング割当確認画像」も生成します。
2. [デコード＆デバッグ]: 保存された画像を読み込み、OpenCVを用いて段階的に解析します。
   解析の各ステップの画像を保存し、コンフィグのマッピング情報に従って
   最終的な文字列の復元（デコード）を行います。
"""

import os
import cv2
import numpy as np

# ==========================================
# donutcodeのインポート
from donutcode import Encoder, Decoder

# ==========================================

# ==========================================
# テスト用設定 
# ==========================================
CONFIG_TYPE = "D-27-13"
TEST_MESSAGE = "134.2335,133.6387"
OUTPUT_DIR = "sample-result"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "test_fresh_donut.png")
DEBUG_MAPPING_IMAGE = os.path.join(OUTPUT_DIR, "00_mapping_debug.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# メイン処理
# ==========================================
def main():
    print("===== DonutCode 統合テスト開始 =====")
    
    # ---------------------------------------------------------
    # 1. 画像の生成 (エンコード)
    # ---------------------------------------------------------
    print(f"\n[1] エンコードを実行します (メッセージ: '{TEST_MESSAGE}')")
    try:
        # コンフィグを指定してエンコーダを初期化
        encoder = Encoder(config_type=CONFIG_TYPE)
        
        # デバッグ用マッピング画像の生成
        encoder.save_mapping_debug_image(DEBUG_MAPPING_IMAGE, scale=20, padding=20)
        print(f" -> [デバッグ] マッピング確認画像を生成しました: {DEBUG_MAPPING_IMAGE}")

        # 本番のエンコード画像生成
        matrix = encoder.encode(TEST_MESSAGE)
        encoder.save_image(matrix, OUTPUT_IMAGE, scale=15, hole_color="#ffebee")
        print(f" -> 新しいコード画像の生成に成功しました: {OUTPUT_IMAGE}")
        
    except Exception as e:
        print(f"❌ エンコード中にエラーが発生しました: {e}")
        return

    # ---------------------------------------------------------
    # 2. 画像の解析・デコード
    # ---------------------------------------------------------
    print(f"\n[2] デコードを実行します ('{OUTPUT_IMAGE}' を読み込み)")
    try:
        # コンフィグを指定してデコーダを初期化
        decoder = Decoder(config_type=CONFIG_TYPE)
        
        # 内部で VisionProcessor による補正からデータ復元まで一気通貫で行われます
        result = decoder.decode_image(OUTPUT_IMAGE)
        
        if result:
            print(f"\n🎉 最終デコード成功！ 復元されたデータ: 【 {result} 】")
        else:
            print("\n❌ デコード失敗 (データが見つからないか破損しています)")
            
    except Exception as e:
        print(f"\n❌ デコード処理中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()