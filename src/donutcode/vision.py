# vision.py
import cv2
import numpy as np

class VisionProcessor:
    def __init__(self, config):
        self.config = config  # configを受け取る
        self.grid_size = self.config.GRID_SIZE
        
    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 【修正点1】キャンバスサイズに依存する適応的閾値をやめ、大津の二値化を採用
        # 背景とコードのコントラストから自動的に最適な閾値を計算します
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        finder_centers = []
        if hierarchy is not None:
            for i in range(len(contours)):
                c1_idx = hierarchy[0][i][2]
                if c1_idx != -1:
                    c2_idx = hierarchy[0][c1_idx][2]
                    if c2_idx != -1:
                        M = cv2.moments(contours[i])
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            if not any(np.hypot(cx - ex, cy - ey) < 10 for ex, ey in finder_centers):
                                finder_centers.append([cx, cy])

        if len(finder_centers) < 3:
            print(" -> [Vision] ファインダパタンが3つ見つかりません。全体輪郭モードで補正します。")
            return self._fallback_crop(img, thresh)

        print(f" -> [Vision] {len(finder_centers)}個のファインダ候補を発見。透視変換を実行します。")
        
        pts = np.array(finder_centers[:3], dtype=np.float32)

        # 3点の距離を計算してTL, TR, BLを特定
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d02 = np.linalg.norm(pts[0] - pts[2])
        
        max_d = max(d01, d12, d02)
        if max_d == d01:   tl_idx, tr_idx, bl_idx = 2, 0, 1
        elif max_d == d12: tl_idx, tr_idx, bl_idx = 0, 1, 2
        else:              tl_idx, tr_idx, bl_idx = 1, 0, 2
        
        TL = pts[tl_idx]
        
        cross = np.cross(pts[tr_idx] - TL, pts[bl_idx] - TL)
        if cross < 0:
            TR, BL = pts[tr_idx], pts[bl_idx]
        else:
            TR, BL = pts[bl_idx], pts[tr_idx]

        # =========================================================
        # 1. ターゲットサイズとセルサイズの動的計算
        # =========================================================
        dist_x_px = np.linalg.norm(TR - TL)
        dist_y_px = np.linalg.norm(BL - TL)
        
        # TL中心からTR中心までは (GRID_SIZE - 7) セル離れている
        cells_between = self.grid_size - 7
        
        # X軸とY軸の平均を取って、1セルあたりの正確なピクセル数を計算
        cell_size_px = ((dist_x_px + dist_y_px) / 2.0) / cells_between
        
        # 画像から逆算した最適な全体ピクセルサイズ
        target_side = int(cell_size_px * self.grid_size)
        print(f" -> [Vision] 動的サイズ計算: 1セル={cell_size_px:.1f}px, 全体出力={target_side}x{target_side}px")

        # =========================================================
        # 2. アライメントパターンの探索（高精度化）
        # =========================================================
        estimated_BR = TR + BL - TL
        actual_alignment = None
        
        if hasattr(self.config, 'ALIGNMENT_POS') and self.config.ALIGNMENT_POS:
            # 探索半径をセルサイズから決定（約4セル分の範囲を探す）
            search_radius = cell_size_px * 4 
            
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    # estimated_BR付近にあり、かつ他のファインダパタンではないものを探す
                    if np.hypot(cx - estimated_BR[0], cy - estimated_BR[1]) < search_radius:
                        # ファインダ自身を誤検知しないよう除外
                        if (np.hypot(cx - TL[0], cy - TL[1]) > search_radius and
                            np.hypot(cx - TR[0], cy - TR[1]) > search_radius and
                            np.hypot(cx - BL[0], cy - BL[1]) > search_radius):
                            actual_alignment = [cx, cy]
                            break

        # =========================================================
        # 3. 透視変換（動的セルサイズを使ってマッピング）
        # =========================================================
        c = cell_size_px
        gs = self.grid_size

        if actual_alignment:
            print(" -> [Vision] アライメントパターンを発見。4点高精度補正を行います。")
            ax, ay = self.config.ALIGNMENT_POS
            # アライメント中心は5x5なので、左上座標 + 2.5セル
            a_center_x = ax + 2.5
            a_center_y = ay + 2.5
            
            src_pts = np.float32([TL, TR, BL, actual_alignment])
            dst_pts = np.float32([
                [3.5 * c, 3.5 * c],
                [(gs - 3.5) * c, 3.5 * c],
                [3.5 * c, (gs - 3.5) * c],
                [a_center_x * c, a_center_y * c]
            ])
        else:
            print(" -> [Vision] アライメント未検出。3点+推測座標で補正します。")
            src_pts = np.float32([TL, TR, BL, estimated_BR])
            dst_pts = np.float32([
                [3.5 * c, 3.5 * c],
                [(gs - 3.5) * c, 3.5 * c],
                [3.5 * c, (gs - 3.5) * c],
                [(gs - 3.5) * c, (gs - 3.5) * c]
            ])

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M_persp, (target_side, target_side))

        return warped

    def _fallback_crop(self, img, thresh):
        print(" -> [Vision] フォールバック: 四角形近似による透視変換を試みます。")
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return img
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 輪郭を多角形に近似（周囲長の数%の精度で直線を当てはめる）
        epsilon = 0.05 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        min_dim = min(img.shape[0], img.shape[1])
        target_side = min_dim

        # もし綺麗に4つの頂点（四角形）が見つかったら、それを透視変換する
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            # 頂点を左上、右上、右下、左下に並び替える処理
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            rect_pts = np.zeros((4, 2), dtype="float32")
            rect_pts[0] = pts[np.argmin(s)]       # 左上
            rect_pts[2] = pts[np.argmax(s)]       # 右下
            rect_pts[1] = pts[np.argmin(diff)]    # 右上
            rect_pts[3] = pts[np.argmax(diff)]    # 左下

            dst_pts = np.float32([[0, 0], [target_side, 0], [target_side, target_side], [0, target_side]])
            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            return cv2.warpPerspective(img, M, (target_side, target_side))
            
        else:
            # 四角形に見立てられなかった場合の最終手段（現状のまま）
            print(" -> [Vision] 四角形の特定に失敗。単純な矩形切り抜きを行います。")
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            
            s = box.sum(axis=1)
            diff = np.diff(box, axis=1)
            rect_pts = np.zeros((4, 2), dtype="float32")
            rect_pts[0] = box[np.argmin(s)]
            rect_pts[2] = box[np.argmax(s)]
            rect_pts[1] = box[np.argmin(diff)]
            rect_pts[3] = box[np.argmax(diff)]
            
            dst_pts = np.float32([[0, 0], [target_side, 0], [target_side, target_side], [0, target_side]])
            M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
            return cv2.warpPerspective(img, M, (target_side, target_side))