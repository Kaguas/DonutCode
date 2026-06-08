"""
DonutCode リアルタイムスキャナー (2段階パースペクティブ補正・完全版)
"""

import cv2
import numpy as np
import math
import donutcode

# ==========================================
# 読み取り設定 (v1.0.0 パターンAに準拠)
# ==========================================
GRID_SIZE = 25
HOLE_RECT = (7, 7, 11, 11)

def order_finders(finders):
    """3つのファインダーから TL(左上), TR(右上), BL(左下) を特定する"""
    pts = [f[0:2] for f in finders]
    d01 = math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
    d12 = math.hypot(pts[1][0] - pts[2][0], pts[1][1] - pts[2][1])
    d20 = math.hypot(pts[2][0] - pts[0][0], pts[2][1] - pts[0][1])

    max_d = max(d01, d12, d20)
    if max_d == d12:
        tl, a, b = 0, 1, 2
    elif max_d == d20:
        tl, a, b = 1, 2, 0
    else:
        tl, a, b = 2, 0, 1

    TL, A, B = pts[tl], pts[a], pts[b]
    
    # 外積で TR と BL を判別
    cross = (A[0] - TL[0]) * (B[1] - TL[1]) - (A[1] - TL[1]) * (B[0] - TL[0])
    if cross > 0:
        return TL, A, B
    else:
        return TL, B, A

def main():
    print("===== DonutCode リアルタイムスキャナー (アンカー補正搭載) 起動 =====")
    decoder = donutcode.Decoder(grid_size=GRID_SIZE, hole_rect=HOLE_RECT)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ エラー: カメラにアクセスできませんでした。")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. 適応的二値化（照明ムラ・影・黒背景を克服）
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5
        )

        # 2. 階層構造で3隅のファインダーを検出
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        finders = []
        if hierarchy is not None:
            for i, cnt in enumerate(contours):
                child_idx = hierarchy[0][i][2]
                if child_idx != -1:
                    grandchild_idx = hierarchy[0][child_idx][2]
                    if grandchild_idx != -1:
                        a1 = cv2.contourArea(cnt)
                        a2 = cv2.contourArea(contours[child_idx])
                        a3 = cv2.contourArea(contours[grandchild_idx])
                        
                        if a1 > 50: 
                            if 0.2 < (a2 / a1) < 0.8 and 0.05 < (a3 / a1) < 0.4:
                                M = cv2.moments(cnt)
                                if M["m00"] != 0:
                                    cx = M["m10"] / M["m00"]
                                    cy = M["m01"] / M["m00"]
                                    finders.append((cx, cy, cnt))
        
        finders = sorted(finders, key=lambda f: cv2.contourArea(f[2]), reverse=True)[:3]

        if len(finders) == 3:
            TL, TR, BL = order_finders(finders)
            
            # 【第1段階】平行四辺形で仮推測し、一旦正面化する
            BR_guess = (TR[0] + BL[0] - TL[0], TR[1] + BL[1] - TL[1])
            src_pts = np.float32([TL, TR, BR_guess, BL])
            
            side_length = 500 # サンプリング精度を上げるため 500x500 に
            cell = side_length / GRID_SIZE
            
            # ファインダーの中心は端から3.5マスの位置
            dst_pts = np.float32([
                [3.5 * cell, 3.5 * cell],
                [(GRID_SIZE - 3.5) * cell, 3.5 * cell],
                [(GRID_SIZE - 3.5) * cell, (GRID_SIZE - 3.5) * cell],
                [3.5 * cell, (GRID_SIZE - 3.5) * cell]
            ])
            
            M_guess = cv2.getPerspectiveTransform(src_pts, dst_pts)
            # 二値化画像を仮正面化する
            warped_thresh = cv2.warpPerspective(thresh, M_guess, (side_length, side_length))
            
            # 【第2段階】仮正面化された画像から、右下アンカー(3x3)の真の中心を探す
            # アンカーの理論上の中心は (GRID_SIZE - 1.5, GRID_SIZE - 1.5) マス目
            anchor_ideal_x = (GRID_SIZE - 1.5) * cell
            anchor_ideal_y = (GRID_SIZE - 1.5) * cell
            
            # パースのズレを考慮し、半径3マス分のエリアを探索
            search_radius = int(3.0 * cell)
            roi_x1 = max(0, int(anchor_ideal_x - search_radius))
            roi_y1 = max(0, int(anchor_ideal_y - search_radius))
            roi_x2 = min(side_length, int(anchor_ideal_x + search_radius))
            roi_y2 = min(side_length, int(anchor_ideal_y + search_radius))
            
            roi = warped_thresh[roi_y1:roi_y2, roi_x1:roi_x2]
            
            anchor_real_x, anchor_real_y = anchor_ideal_x, anchor_ideal_y
            if roi.size > 0:
                # ROI内の輪郭を探し、最も大きい黒の塊を真のアンカーとみなす
                roi_contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if roi_contours:
                    largest_cnt = max(roi_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_cnt) > (cell * cell): # ノイズ弾き
                        M_roi = cv2.moments(largest_cnt)
                        if M_roi["m00"] != 0:
                            cx = M_roi["m10"] / M_roi["m00"]
                            cy = M_roi["m01"] / M_roi["m00"]
                            anchor_real_x = roi_x1 + cx
                            anchor_real_y = roi_y1 + cy

            # 真のアンカー座標を使って、第2の補正行列を作る
            src_pts2 = np.float32([
                [3.5 * cell, 3.5 * cell],
                [(GRID_SIZE - 3.5) * cell, 3.5 * cell],
                [anchor_real_x, anchor_real_y],
                [3.5 * cell, (GRID_SIZE - 3.5) * cell]
            ])
            dst_pts2 = np.float32([
                [3.5 * cell, 3.5 * cell],
                [(GRID_SIZE - 3.5) * cell, 3.5 * cell],
                [(GRID_SIZE - 1.5) * cell, (GRID_SIZE - 1.5) * cell],
                [3.5 * cell, (GRID_SIZE - 3.5) * cell]
            ])
            
            M2 = cv2.getPerspectiveTransform(src_pts2, dst_pts2)
            
            # 【最終補正】元画像(frame)を M_guess → M2 の順で重ねて変形
            warped_color_guess = cv2.warpPerspective(frame, M_guess, (side_length, side_length))
            final_cropped = cv2.warpPerspective(warped_color_guess, M2, (side_length, side_length))
            
            try:
                # 歪みのない完璧な画像でデコード処理
                bit_map = decoder._extract_bit_map(final_cropped)
                decoded_text = decoder._decode_from_bit_map(bit_map)
                
                if decoded_text:
                    # 成功時：AR描画
                    # 最終的な枠線を、元のカメラ映像の座標に逆変換して描画する
                    M_inv = np.linalg.inv(M2 @ M_guess)
                    real_corners = np.float32([[0,0], [side_length,0], [side_length,side_length], [0,side_length]])
                    real_corners = np.array([real_corners])
                    
                    frame_poly = cv2.perspectiveTransform(real_corners, M_inv)[0].astype(np.int32)
                    cv2.polylines(frame, [frame_poly], True, (0, 255, 0), 3)
                    
                    text_x, text_y = frame_poly[0][0], frame_poly[0][1] - 15
                    cv2.rectangle(frame, (text_x, text_y - 30), (text_x + 350, text_y + 10), (0, 0, 0), -1)
                    cv2.putText(frame, decoded_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception:
                pass

        cv2.imshow("DonutCode Scanner", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("スキャナーを終了しました。")

if __name__ == "__main__":
    main()