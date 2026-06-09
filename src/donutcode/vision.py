# vision.py
import cv2
import numpy as np

"""DonutCodeの画像処理モジュール
このモジュールは、DonutCodeの画像からコード領域を検出して、透視変換を行うためのクラスを提供します。
主な機能:
- ファインダパタンの検出と位置特定
- 透視変換によるコード領域の正規化
- フォールバックロジックによる頑健な検出
"""

class VisionProcessor:
    def __init__(self, config=None, target_side=500):
        self.config = config
        # configが渡されなかった場合のフォールバック
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        """階層構造からファインダ候補を抽出する関数（正方形判定付き）"""
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
                    
                    # 極端に小さいノイズは無条件で弾く
                    if area < 50:
                        continue
                    
                    # 形状判定（アスペクト比）
                    rect = cv2.minAreaRect(c)
                    w, h = rect[1]
                    if w == 0 or h == 0:
                        continue
                    aspect_ratio = max(w, h) / min(w, h)

                    # モードによる条件分岐
                    is_valid_shape = False
                    if strict_mode:
                        # 厳格モード: ほぼ正方形 (1.0〜1.3倍以内)
                        if aspect_ratio < 1.3:
                            is_valid_shape = True
                    else:
                        # 緩和モード: 多少歪んでいても許容 (1.0〜2.0倍以内)
                        if aspect_ratio < 2.0:
                            is_valid_shape = True

                    if is_valid_shape:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            # 重複検出の防止
                            if not any(np.hypot(cx - ex, cy - ey) < 15 for ex, ey in centers):
                                centers.append([cx, cy])
                                valid_contours.append(c)
                                
        return centers, valid_contours

    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 二値化 (SAM環境・ごちゃごちゃ背景で実績のあるパラメータ)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 5
        )

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ==========================================
        # 3段構えのフォールバック・ロジック (Tier 1 ~ 3)
        # ==========================================
        print(" -> [Vision] [Tier 1] 厳格なファインダ探索を実行中...")
        centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
        
        if len(centers) >= 3:
            print(f" -> [Vision] [Tier 1] 成功！ {len(centers)}個のファインダを発見しました。")
            pts = np.array(centers[:3], dtype=np.float32)
        else:
            print(" -> [Vision] [Tier 1] 失敗。")
            print(" -> [Vision] [Tier 2] 条件を緩和してファインダ探索を実行中...")
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
            
            if len(centers) >= 3:
                print(f" -> [Vision] [Tier 2] 成功！ {len(centers)}個のファインダを発見しました。")
                pts = np.array(centers[:3], dtype=np.float32)
            else:
                print(" -> [Vision] [Tier 2] 失敗。ファインダが3つ見つかりません。")
                return self._fallback_crop(img, thresh)

        # --- Tier 1 か Tier 2 で成功した場合の幾何学計算 ---
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
        
        # 探索のためにセルサイズを概算
        dist_x_px = np.linalg.norm(TR - TL)
        dist_y_px = np.linalg.norm(BL - TL)
        cells_between = self.grid_size - 7
        cell_size_px = ((dist_x_px + dist_y_px) / 2.0) / cells_between

        if self.config and hasattr(self.config, 'ALIGNMENT_POS') and self.config.ALIGNMENT_POS:
            search_radius = cell_size_px * 4 
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    if np.hypot(cx - estimated_BR[0], cy - estimated_BR[1]) < search_radius:
                        # ファインダ自身の誤検知を除外
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
            print(" -> [Vision] アライメントパターンを発見。4点高精度補正を行います。")
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
            print(" -> [Vision] アライメント未検出。3点+推測座標で補正します。")
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
        print(" -> [Vision] [Tier 3] 全体の矩形探索モードを実行します。")
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