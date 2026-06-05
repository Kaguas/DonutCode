"""
DonutCode 統合テスト＆デバッグスクリプト

【概要】
このスクリプトは、DonutCodeライブラリの動作確認を行うための総合テストツールです。
以下の2つの処理を連続して実行します。

1. [エンコード]: 指定された設定（サイズ・穴の位置・エラー訂正・テキスト）から
   DonutCodeを生成し、画像として保存します。
2. [デコード＆デバッグ]: 保存された画像を読み込み、donutcode.Decoderクラスを用いて
   段階的に解析します。解析の各ステップの画像を保存し、最終的な文字列の復元を行います。

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
GRID_SIZE = 27
HOLE_RECT = (7, 7, 13, 13)
ECC_BYTES = 24

# 小数点4桁（11メートル精度）
TEST_MESSAGE = "134.2335,133.6387"
OUTPUT_DIR = "sample-result"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "test_fresh_donut.png")

# 出力先フォルダの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# メイン処理
# ==========================================
def main():
    print("===== DonutCode 統合テスト開始 =====")
    
    # ---------------------------------------------------------
    # 1. 画像の生成 (エンコード)
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 2. 画像の解析・デコード (デバッグ出力付き)
    # ---------------------------------------------------------
    print(f"\n[2] デコード解析を開始します ('{OUTPUT_IMAGE}' を読み込み)")
    
    src_img = cv2.imread(OUTPUT_IMAGE)
    if src_img is None:
        print(f"❌ エラー: 画像 '{OUTPUT_IMAGE}' が読み込めません。")
        return

    # ★ リファクタリングしたDecoderの呼び出し ★
    decoder = donutcode.Decoder(
        grid_size=GRID_SIZE, 
        hole_rect=HOLE_RECT, 
        ecc_bytes=ECC_BYTES
    )

    print(" -> 画像の読み込みに成功しました。デコード処理を開始します...")
    direct_decode = decoder.decode_image(OUTPUT_IMAGE)
    print(f" -> decode_image() の直接呼び出し結果: {direct_decode}")


    # --- STEP A: 領域検出 ---
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh_inv)
    if coords is None:
        print("コードの領域を検出できませんでした。")
        return

    rect = cv2.minAreaRect(coords)
    box = np.int32(cv2.boxPoints(rect))

    # 検出結果の保存 (緑枠)
    debug_img1 = src_img.copy()
    cv2.drawContours(debug_img1, [box], 0, (0, 255, 0), 2)
    out_path1 = os.path.join(OUTPUT_DIR, '01_detected.png')
    cv2.imwrite(out_path1, debug_img1)
    print(f" -> [STEP 1] 領域検出完了: {out_path1}")

    # --- STEP B: 正面補正 (Decoderのメソッドを使用) ---
    side_length = 500
    cropped_img = decoder.warp_to_square(src_img, box, side_length=side_length)
    
    out_path2 = os.path.join(OUTPUT_DIR, '02_cropped.png')
    cv2.imwrite(out_path2, cropped_img)
    print(f" -> [STEP 2] 正面補正完了: {out_path2}")

    # --- STEP C: マス目のサンプリング (Decoderのメソッドを使用) ---
    bit_map = decoder.image_to_bitmap(cropped_img)

    # デバッグ画像用に抽出したビットマップの青・赤点を描画
    debug_img3 = cropped_img.copy()
    cell_size = side_length / GRID_SIZE
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx = int((col + 0.5) * cell_size)
            cy = int((row + 0.5) * cell_size)
            if bit_map[row, col] == 1:
                cv2.circle(debug_img3, (cx, cy), 4, (255, 0, 0), -1) # 青:黒
            else:
                cv2.circle(debug_img3, (cx, cy), 4, (0, 0, 255), -1) # 赤:白
                
    out_path3 = os.path.join(OUTPUT_DIR, '03_sampled.png')
    cv2.imwrite(out_path3, debug_img3)
    print(f" -> [STEP 3] サンプリング完了 (青:黒, 赤:白): {out_path3}")

    # --- STEP D: デコード (Decoderのメソッドを使用) ---
    decoded_text = decoder._decode_from_bit_map(bit_map)

    if decoded_text:
        print(f"\n🎉 最終デコード成功！ 復元されたデータ: 【 {decoded_text} 】")
    else:
        print("\nデコード失敗 (データ破損)")

    # ==========================================
    # (参考) 実運用環境での呼び出し方
    # ==========================================
    print("\n---")
    print("[参考] 実際の運用環境(デバッグ画像が不要な場合)では、以下の1行で完結します:")
    results = decoder.decode_image(OUTPUT_IMAGE)
    print(f"decoder.decode_image() の戻り値: {results}")

if __name__ == "__main__":
    main()