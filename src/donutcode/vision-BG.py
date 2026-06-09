# vision.py
import cv2
import numpy as np

"""DonutCodeの画像処理モジュール
このモジュールは、DonutCodeの画像からコード領域を検出して、透視変換を行うためのクラスを提供します。
主な機能:
- ファインダパタンの検出と位置特定
- 透視変換によるコード領域の正規化
- pyzbarはラインでの走査をしているがPYTHONの実行速度的にここで実装するのは無理があるので、
- 複数画像を並列処理することでロバスト性を上げる。
- フォールバックロジックによる頑健な検出
"""

class VisionProcessor:
    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    # =======================================================
    #  1Dスキャンライン (ZBar風) の実験的実装モジュール
    # =======================================================
    def _check_ratio(self, state_count):
        """ピクセル幅が 1:1:3:1:1 のファインダパターン比率になっているか判定"""
        total_width = sum(state_count)
        if total_width < 7:
            return False
        
        module_size = total_width / 7.0
        # 小さい画像（ノイズ多い）を想定し、理論値から70%の誤差を許容
        max_variance = module_size * 0.7 

        return (abs(module_size - state_count[0]) < max_variance and
                abs(module_size - state_count[1]) < max_variance and
                abs(3.0 * module_size - state_count[2]) < 3.0 * max_variance and
                abs(module_size - state_count[3]) < max_variance and
                abs(module_size - state_count[4]) < max_variance)

    def _cross_check_vertical(self, thresh, center_x, center_y):
        """水平方向で見つけた中心を、垂直方向にも走査してクロスチェックする"""
        h, w = thresh.shape
        state_count = [0, 0, 0, 0, 0]
        
        # 1. 上へ向かって走査
        y = center_y
        while y >= 0 and thresh[y, center_x] == 255: # 中心黒(3)の上半分
            state_count[2] += 1
            y -= 1
        if y < 0: return False
        while y >= 0 and thresh[y, center_x] == 0:   # 内白(1)
            state_count[1] += 1
            y -= 1
        if y < 0: return False
        while y >= 0 and thresh[y, center_x] == 255: # 外黒(1)
            state_count[0] += 1
            y -= 1
            
        # 2. 下へ向かって走査
        y = center_y + 1
        while y < h and thresh[y, center_x] == 255: # 中心黒(3)の下半分
            state_count[2] += 1
            y += 1
        if y == h: return False
        while y < h and thresh[y, center_x] == 0:   # 内白(1)
            state_count[3] += 1
            y += 1
        if y == h: return False
        while y < h and thresh[y, center_x] == 255: # 外黒(1)
            state_count[4] += 1
            y += 1
            
        return self._check_ratio(state_count)

    def _cluster_centers(self, centers):
        """複数の走査線で同じファインダが何度もヒットするため、近接する座標を1つに統合する"""
        clusters = []
        for cx, cy in centers:
            found = False
            for cluster in clusters:
                dist = np.hypot(cluster['x'] - cx, cluster['y'] - cy)
                if dist < 15: # 15ピクセル以内のヒットは同一ファインダとみなす
                    cluster['x'] = (cluster['x'] * cluster['count'] + cx) / (cluster['count'] + 1)
                    cluster['y'] = (cluster['y'] * cluster['count'] + cy) / (cluster['count'] + 1)
                    cluster['count'] += 1
                    found = True
                    break
            if not found:
                clusters.append({'x': cx, 'y': cy, 'count': 1})
        
        # 複数回ヒットした（確度が高い）クラスタだけを返すのが理想だが、
        # 今回は低解像度実験のため1回でもヒットすれば候補に入れる
        return [[c['x'], c['y']] for c in clusters]

    def _scan_finder_patterns_1d(self, thresh):
        """画像全体に水平の走査線を引き、ファインダパターンを探索するメイン関数"""
        h, w = thresh.shape
        centers = []
        skip = max(1, h // 200) # 画像が大きければ数行飛ばして高速化
        
        for y in range(0, h, skip):
            state_count = [0, 0, 0, 0, 0]
            current_state = 0
            
            for x in range(w):
                is_black = (thresh[y, x] == 255) # INV二値化なので255が「黒(インク)」
                
                if is_black:
                    if current_state % 2 == 1: # 白のカウント中だったなら
                        current_state += 1     # 次の黒へ状態遷移
                    state_count[current_state] += 1
                else: # 白ピクセルの場合
                    if current_state == 0 and state_count[0] == 0:
                        continue # 最初の黒が来るまでは白を無視する
                    
                    if current_state % 2 == 0: # 黒のカウント中だったなら
                        if current_state == 4:
                            # 1:1:3:1:1 のパターン(黒-白-黒-白-黒)が完成したかチェック
                            if self._check_ratio(state_count):
                                # パターンの水平中心X座標を計算
                                center_x = x - state_count[4] - state_count[3] - state_count[2] / 2.0
                                # 垂直方向にも線を引いてクロスチェック（偽陽性を防ぐ）
                                if self._cross_check_vertical(thresh, int(center_x), y):
                                    centers.append((center_x, y))
                            
                            # 状態をリセットして再開
                            state_count = [0, 0, 0, 0, 0]
                            current_state = 0
                        else:
                            current_state += 1 # 次の白へ状態遷移
                            state_count[current_state] += 1
                    else:
                        state_count[current_state] += 1
                        
        return self._cluster_centers(centers)
    # =======================================================


    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        """階層構造とアスペクト比からファインダ候補を抽出する (2D輪郭アプローチ)"""
        centers = []
        valid_contours = []
        if hierarchy is None: return centers, valid_contours

        for i in range(len(contours)):
            c1_idx = hierarchy[0][i][2]
            if c1_idx != -1:
                c2_idx = hierarchy[0][c1_idx][2]
                if c2_idx != -1:
                    c = contours[i]
                    if cv2.contourArea(c) < 50: continue
                    
                    rect = cv2.minAreaRect(c)
                    w, h = rect[1]
                    if w == 0 or h == 0: continue
                    aspect_ratio = max(w, h) / min(w, h)

                    is_valid = (aspect_ratio < 1.3) if strict_mode else (aspect_ratio < 2.0)
                    if is_valid:
                        M = cv2.moments(c)
                        if M["m00"] != 0:
                            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                            if not any(np.hypot(cx - ex, cy - ey) < 15 for ex, ey in centers):
                                centers.append([cx, cy])
                                valid_contours.append(c)
        return centers, valid_contours


    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 多重閾値
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresholds = [("Adaptive-21", thresh1), ("Adaptive-41", thresh2), ("OTSU", thresh3)]

        best_centers = None
        best_contours = None
        
        # --- [Tier 1 & 2] 既存の2D輪郭アプローチ ---
        for name, thresh in thresholds:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
            if len(centers) >= 3:
                best_centers, best_contours = centers, contours
                break
                
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
            if len(centers) >= 3:
                best_centers, best_contours = centers, contours
                break

        # --- [Tier 2.5] 1Dスキャンライン (実験的実装) ---
        # 輪郭アプローチで全滅した場合のみ発動します
        if not best_centers or len(best_centers) < 3:
            print(" -> [Vision] [Tier 2.5] 輪郭抽出に失敗。1Dスキャンライン探索(Python版)を実行します...")
            for name, thresh in thresholds:
                centers = self._scan_finder_patterns_1d(thresh)
                if len(centers) >= 3:
                    print(f" -> [Vision] [Tier 2.5] 成功！ {name} で {len(centers)}個のファインダを発見。")
                    best_centers = centers
                    best_contours = None # 1Dスキャンでは輪郭オブジェクトが存在しないためNone
                    break

        # --- [Tier 3] 最終フォールバック ---
        if not best_centers or len(best_centers) < 3:
            return self._fallback_crop(img, thresh1)

        # === 幾何学計算 ===
        pts = np.array(best_centers[:3], dtype=np.float32)
        d01 = np.linalg.norm(pts[0] - pts[1])
        d12 = np.linalg.norm(pts[1] - pts[2])
        d02 = np.linalg.norm(pts[0] - pts[2])
        
        max_d = max(d01, d12, d02)
        if max_d == d01:   tl_idx, tr_idx, bl_idx = 2, 0, 1
        elif max_d == d12: tl_idx, tr_idx, bl_idx = 0, 1, 2
        else:              tl_idx, tr_idx, bl_idx = 1, 0, 2
        
        TL = pts[tl_idx]
        if np.cross(pts[tr_idx] - TL, pts[bl_idx] - TL) < 0:
            TR, BL = pts[tr_idx], pts[bl_idx]
        else:
            TR, BL = pts[bl_idx], pts[tr_idx]

        estimated_BR = TR + BL - TL

        # === アライメント補正 ===
        actual_alignment = None
        if best_contours and self.config and hasattr(self.config, 'ALIGNMENT_POS') and self.config.ALIGNMENT_POS:
            # 1Dスキャン経由(best_contoursがNone)の場合はエラーになるためスキップし、推測BRを使用します
            dist_x_px = np.linalg.norm(TR - TL)
            dist_y_px = np.linalg.norm(BL - TL)
            cell_size_px = ((dist_x_px + dist_y_px) / 2.0) / (self.grid_size - 7)
            search_radius = cell_size_px * 4 

            for c in best_contours:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    if np.hypot(cx - estimated_BR[0], cy - estimated_BR[1]) < search_radius:
                        if (np.hypot(cx - TL[0], cy - TL[1]) > search_radius and
                            np.hypot(cx - TR[0], cy - TR[1]) > search_radius and
                            np.hypot(cx - BL[0], cy - BL[1]) > search_radius):
                            actual_alignment = [cx, cy]
                            break

        # === 透視変換 ===
        gs, ts = self.grid_size, self.target_side
        if actual_alignment:
            ax, ay = self.config.ALIGNMENT_POS
            a_center_x, a_center_y = ax + 2.5, ay + 2.5
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
        return cv2.warpPerspective(img, M_persp, (ts, ts))


    def _fallback_crop(self, img, thresh):
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ts = self.target_side
        if not contours: return cv2.resize(img, (ts, ts))
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(rect)
        s, diff = box.sum(axis=1), np.diff(box, axis=1)
        rect_pts = np.float32([box[np.argmin(s)], box[np.argmin(diff)], box[np.argmax(s)], box[np.argmax(diff)]])
        dst_pts = np.float32([[0, 0], [ts, 0], [ts, ts], [0, ts]])
        return cv2.warpPerspective(img, cv2.getPerspectiveTransform(rect_pts, dst_pts), (ts, ts))