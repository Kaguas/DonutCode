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
#無駄がないパターンその1
GRID_SIZE = 25
HOLE_RECT = (7, 7, 11, 11)
ECC_BYTES = 22

"""
#無駄がないパターンその2
GRID_SIZE = 27
HOLE_RECT = (7, 7, 13, 13)
ECC_BYTES = 29
"""
# 小数点4桁（11メートル精度）にしています。
TEST_MESSAGE = "134.2335,133.6387"
OUTPUT_DIR = "sample-result"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "test_fresh_donut.png")

# 出力先フォルダの作成
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# デバッグ用補助関数
# ==========================================
def _is_reserved_pattern(col, row, grid_size):
    # 3隅のファインダーパターン (8x8)
    if 0 <= col < 8 and 0 <= row < 8: return True
    if grid_size - 8 <= col < grid_size and 0 <= row < 8: return True
    if 0 <= col < 8 and grid_size - 8 <= row < grid_size: return True
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

    # --- STEP A: 領域検出 ---
    # 白黒反転して黒いピクセル群（コード部分）を抽出
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    _, thresh_inv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh_inv)
    if coords is None:
        print("❌ コードの領域を検出できませんでした。")
        return

    rect = cv2.minAreaRect(coords)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # 検出結果の保存 (緑枠)
    debug_img1 = src_img.copy()
    cv2.drawContours(debug_img1, [box], 0, (0, 255, 0), 2)
    out_path1 = os.path.join(OUTPUT_DIR, '01_detected.png')
    cv2.imwrite(out_path1, debug_img1)
    print(f" -> [STEP 1] 領域検出完了: {out_path1}")

    # --- STEP B: 正面補正 (透視変換) ---
    pts1 = _order_points(box.astype("float32"))
    side_length = 500  # 解析用に大きめの正方形に補正
    pts2 = np.float32([[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    cropped_img = cv2.warpPerspective(src_img, M, (side_length, side_length))

    # 補正画像の保存
    out_path2 = os.path.join(OUTPUT_DIR, '02_cropped.png')
    cv2.imwrite(out_path2, cropped_img)
    print(f" -> [STEP 2] 正面補正完了: {out_path2}")

    # --- STEP C: マス目のサンプリング ---
    gray_crop = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh_crop = cv2.threshold(gray_crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cell_size = side_length / GRID_SIZE
    bit_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    debug_img3 = cropped_img.copy()

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            center_x = int((col + 0.5) * cell_size)
            center_y = int((row + 0.5) * cell_size)

            if thresh_crop[center_y, center_x] < 128:  # 黒判定
                bit_map[row, col] = 1
                cv2.circle(debug_img3, (center_x, center_y), 4, (255, 0, 0), -1) # 青点
            else:  # 白判定
                bit_map[row, col] = 0
                cv2.circle(debug_img3, (center_x, center_y), 4, (0, 0, 255), -1) # 赤点

    # サンプリング結果の保存
    out_path3 = os.path.join(OUTPUT_DIR, '03_sampled.png')
    cv2.imwrite(out_path3, debug_img3)
    print(f" -> [STEP 3] サンプリング完了 (青:黒, 赤:白): {out_path3}")

    # --- STEP D: デコード (データの復元) ---
    bit_stream = ""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if _is_reserved_pattern(col, row, GRID_SIZE): continue
            if _is_hole(col, row, HOLE_RECT): continue
            bit_stream += str(bit_map[row, col])

    byte_list = bytearray()
    for i in range(0, len(bit_stream), 8):
        byte_str = bit_stream[i:i+8]
        if len(byte_str) < 8: break
        
        byte_val = int(byte_str, 2)
        if byte_val == 0:  # 終端(Null)検知
            break
        byte_list.append(byte_val)

    try:
        decoded_text = byte_list.decode('utf-8')
        print(f"\n🎉 最終デコード成功！ 復元されたデータ: 【 {decoded_text} 】")
    except Exception as e:
        print(f"\n❌ デコード失敗 (データ破損): {e}")

if __name__ == "__main__":
    main()