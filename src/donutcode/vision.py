# vision.py
import cv2
import numpy as np

"""DonutCodeの画像処理モジュール
このモジュールは、DonutCodeの画像からコード領域を検出して、透視変換を行うためのクラスを提供します。
主な機能:
- ファインダパタンの検出と位置特定
- 透視変換によるコード領域の正規化
- pyzbarはラインでの走査をしているがPYTHONの実行速度的にここで実装するのは無理があるので、複数画像を並列処理することでロバスト性を上げる。
- フォールバックロジックによる頑健な検出
"""

class VisionProcessor:
    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        """階層構造とアスペクト比からファインダ候補を抽出する"""
        centers = []
        valid_contours = []
        
        if hierarchy is None:
            return centers, valid_contours

        for i in range(len(contours)):
            c1_idx = hierarchy[0][i][2]
            if c1_idx != -1:
                c2_idx = hierarchy[0][c1_idx][2]
                if c2_idx != -1:
                    c = contours[i]
                    area = cv2.contourArea(c)
                    
                    if area < 50:
                        continue
                    
                    rect = cv2.minAreaRect(c)
                    w, h = rect[1]
                    if w == 0 or h == 0:
                        continue
                    aspect_ratio = max(w, h) / min(w, h)

                    is_valid_shape = False
                    if strict_mode:
                        if aspect_ratio < 1.3:
                            is_valid_shape = True
                    else:
                        if aspect_ratio < 2.0:
                            is_valid_shape = True

                    if is_valid_shape:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            if not any(np.hypot(cx - ex, cy - ey) < 15 for ex, ey in centers):
                                centers.append([cx, cy])
                                valid_contours.append(c)
                                
        return centers, valid_contours

    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ==========================================
        # マルチ・スレッショルド（多重閾値）の生成
        # ==========================================
        # 1. Base Adaptive: 局所的なノイズに強い（これまでの最適解）
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        
        # 2. Wide Adaptive: ブロックサイズが大きく、大きなノイズや影のグラデーションに強い
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        
        # 3. OTSU: 背景と紙のコントラストがはっきりしている環境光下で最強
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        thresholds = [
            ("Adaptive-21", thresh1),
            ("Adaptive-41", thresh2),
            ("OTSU", thresh3)
        ]

        best_centers = None
        best_contours = None
        
        # 3種類の閾値画像を順番にテスト（どれか1つでも通れば勝ち）
        for name, thresh in thresholds:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # [Tier 1] 厳格探索
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
            if len(centers) >= 3:
                # print(f" -> [Vision] [{name}] Tier 1 (厳格) でファインダ発見！") # ログが多すぎる場合はコメントアウト推奨
                best_centers = centers
                best_contours = contours
                break
                
            # [Tier 2] 緩和探索
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
            if len(centers) >= 3:
                # print(f" -> [Vision] [{name}] Tier 2 (緩和) でファインダ発見！")
                best_centers = centers
                best_contours = contours
                break

        # ==========================================
        # 幾何学計算とフォールバック
        # ==========================================
        if not best_centers or len(best_centers) < 3:
            # print(" -> [Vision] 全閾値でファインダ検出に失敗。全体矩形フォールバックへ移行します。")
            return self._fallback_crop(img, thresh1) # 最も標準的なthresh1をベースに切り抜く

        pts = np.array(best_centers[:3], dtype=np.float32)

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

        estimated_BR = TR + BL - TL

        # =========================================================
        # アライメントパターンの探索（高精度化）
        # =========================================================
        actual_alignment = None
        dist_x_px = np.linalg.norm(TR - TL)
        dist_y_px = np.linalg.norm(BL - TL)
        cells_between = self.grid_size - 7
        cell_size_px = ((dist_x_px + dist_y_px) / 2.0) / cells_between

        if self.config and hasattr(self.config, 'ALIGNMENT_POS') and self.config.ALIGNMENT_POS:
            search_radius = cell_size_px * 4 
            for c in best_contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    if np.hypot(cx - estimated_BR[0], cy - estimated_BR[1]) < search_radius:
                        if (np.hypot(cx - TL[0], cy - TL[1]) > search_radius and
                            np.hypot(cx - TR[0], cy - TR[1]) > search_radius and
                            np.hypot(cx - BL[0], cy - BL[1]) > search_radius):
                            actual_alignment = [cx, cy]
                            break

        # =========================================================
        # 透視変換（Warp）
        # =========================================================
        gs = self.grid_size
        ts = self.target_side

        if actual_alignment:
            ax, ay = self.config.ALIGNMENT_POS
            a_center_x = ax + 2.5
            a_center_y = ay + 2.5
            
            src_pts = np.float32([TL, TR, BL, actual_alignment])
            dst_pts = np.float32([
                [3.5 * (ts / gs), 3.5 * (ts / gs)],
                [(gs - 3.5) * (ts / gs), 3.5 * (ts / gs)],
                [3.5 * (ts / gs), (gs - 3.5) * (ts / gs)],
                [a_center_x * (ts / gs), a_center_y * (ts / gs)]
            ])
        else:
            src_pts = np.float32([TL, TR, BL, estimated_BR])
            dst_pts = np.float32([
                [3.5 * (ts / gs), 3.5 * (ts / gs)],
                [(gs - 3.5) * (ts / gs), 3.5 * (ts / gs)],
                [3.5 * (ts / gs), (gs - 3.5) * (ts / gs)],
                [(gs - 3.5) * (ts / gs), (gs - 3.5) * (ts / gs)]
            ])

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M_persp, (ts, ts))

        return warped

    def _fallback_crop(self, img, thresh):
        """【Tier 3】最終手段：全体矩形からの切り出し"""
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ts = self.target_side
        if not contours:
            return cv2.resize(img, (ts, ts))
            
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        
        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)
        rect_pts = np.zeros((4, 2), dtype="float32")
        rect_pts[0] = box[np.argmin(s)]
        rect_pts[2] = box[np.argmax(s)]
        rect_pts[1] = box[np.argmin(diff)]
        rect_pts[3] = box[np.argmax(diff)]
        
        dst_pts = np.float32([[0, 0], [ts, 0], [ts, ts], [0, ts]])
        M_persp = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        return cv2.warpPerspective(img, M_persp, (ts, ts))