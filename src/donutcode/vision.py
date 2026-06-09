# vision.py
import cv2
import numpy as np
import os

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
        total_width = sum(state_count)
        if total_width < 7: return False
        module_size = total_width / 7.0
        max_variance = module_size * 0.7 
        return (abs(module_size - state_count[0]) < max_variance and
                abs(module_size - state_count[1]) < max_variance and
                abs(3.0 * module_size - state_count[2]) < 3.0 * max_variance and
                abs(module_size - state_count[3]) < max_variance and
                abs(module_size - state_count[4]) < max_variance)

    def _cross_check_vertical(self, thresh, center_x, center_y):
        h, w = thresh.shape
        state_count = [0, 0, 0, 0, 0]
        y = center_y
        while y >= 0 and thresh[y, center_x] == 255:
            state_count[2] += 1; y -= 1
        if y < 0: return False
        while y >= 0 and thresh[y, center_x] == 0:
            state_count[1] += 1; y -= 1
        if y < 0: return False
        while y >= 0 and thresh[y, center_x] == 255:
            state_count[0] += 1; y -= 1
            
        y = center_y + 1
        while y < h and thresh[y, center_x] == 255:
            state_count[2] += 1; y += 1
        if y == h: return False
        while y < h and thresh[y, center_x] == 0:
            state_count[3] += 1; y += 1
        if y == h: return False
        while y < h and thresh[y, center_x] == 255:
            state_count[4] += 1; y += 1
        return self._check_ratio(state_count)

    def _cluster_centers(self, centers):
        clusters = []
        for cx, cy in centers:
            found = False
            for cluster in clusters:
                dist = np.hypot(cluster['x'] - cx, cluster['y'] - cy)
                if dist < 15:
                    cluster['x'] = (cluster['x'] * cluster['count'] + cx) / (cluster['count'] + 1)
                    cluster['y'] = (cluster['y'] * cluster['count'] + cy) / (cluster['count'] + 1)
                    cluster['count'] += 1
                    found = True
                    break
            if not found:
                clusters.append({'x': cx, 'y': cy, 'count': 1})
        return [[c['x'], c['y']] for c in clusters]

    def _scan_finder_patterns_1d(self, thresh):
        h, w = thresh.shape
        centers = []
        skip = max(1, h // 200)
        for y in range(0, h, skip):
            state_count = [0, 0, 0, 0, 0]
            current_state = 0
            for x in range(w):
                is_black = (thresh[y, x] == 255)
                if is_black:
                    if current_state % 2 == 1: current_state += 1
                    state_count[current_state] += 1
                else:
                    if current_state == 0 and state_count[0] == 0: continue
                    if current_state % 2 == 0:
                        if current_state == 4:
                            if self._check_ratio(state_count):
                                center_x = x - state_count[4] - state_count[3] - state_count[2] / 2.0
                                if self._cross_check_vertical(thresh, int(center_x), y):
                                    centers.append((center_x, y))
                            state_count = [0, 0, 0, 0, 0]
                            current_state = 0
                        else:
                            current_state += 1
                            state_count[current_state] += 1
                    else:
                        state_count[current_state] += 1
        return self._cluster_centers(centers)

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        centers = []
        valid_contours = []
        if hierarchy is None: return centers, valid_contours

        # ファインダ検出ロジックは旧バージョンの堅牢な実装を維持
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

    # =======================================================
    # process メソッド (デバッグモード追加)
    # =======================================================
    def process(self, img_path: str|np.ndarray, debug_mode=False):
        #!#!# 文字列とnumpy array両方に対応
        if type(img_path) == str:
            img = cv2.imread(img_path)
        else:
            img = img_path
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 5)
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        thresholds = [("Adaptive-21", thresh1), ("Adaptive-41", thresh2), ("OTSU", thresh3)]

        best_centers = None
        best_contours = None
        best_hierarchy = None # 階層構造も保持
        
        for name, thresh in thresholds:
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
            if len(centers) >= 3:
                best_centers, best_contours, best_hierarchy = centers, contours, hierarchy
                break
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
            if len(centers) >= 3:
                best_centers, best_contours, best_hierarchy = centers, contours, hierarchy
                break

        if not best_centers or len(best_centers) < 3:
            for name, thresh in thresholds:
                centers = self._scan_finder_patterns_1d(thresh)
                if len(centers) >= 3:
                    best_centers = centers
                    best_contours = None
                    best_hierarchy = None
                    break

        if not best_centers or len(best_centers) < 3:
            return self._fallback_crop(img, thresh1)

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

        # =======================================================
        # 【修正】アライメントパターンの高精度探索（白四角ベース）
        # =======================================================
        actual_alignment = None
        if best_contours is not None and best_hierarchy is not None and self.config and hasattr(self.config, 'ALIGNMENT_POS') and self.config.ALIGNMENT_POS:
            dist_x_px = np.linalg.norm(TR - TL)
            dist_y_px = np.linalg.norm(BL - TL)
            cell_size_px = ((dist_x_px + dist_y_px) / 2.0) / (self.grid_size - 7)
            search_radius = cell_size_px * 4 

            candidates = []
            for i in range(len(best_contours)):
                c = best_contours[i]
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                dist = np.hypot(cx - estimated_BR[0], cy - estimated_BR[1])
                
                if dist < search_radius:
                    # ファインダパターン自身を除外
                    if (np.hypot(cx - TL[0], cy - TL[1]) > search_radius and
                        np.hypot(cx - TR[0], cy - TR[1]) > search_radius and
                        np.hypot(cx - BL[0], cy - BL[1]) > search_radius):
                        
                        # 形状がほぼ正方形かチェック
                        rect = cv2.minAreaRect(c)
                        w, h = rect[1]
                        if w == 0 or h == 0: continue
                        aspect_ratio = max(w, h) / min(w, h)
                        
                        if aspect_ratio < 1.5:
                            # 階層構造チェック：子要素（内側の黒四角）を持っているか？
                            # 白四角は穴（hole）なので、中に黒四角があれば has_child が True になる
                            has_child = best_hierarchy[0][i][2] != -1
                            candidates.append({
                                'pos': [cx, cy],
                                'dist': dist,
                                'has_child': has_child
                            })
            
            if candidates:
                # 「子要素（内側黒四角）を持つ白四角」を最優先で選ぶ
                with_child = [cand for cand in candidates if cand['has_child']]
                if with_child:
                    best_cand = min(with_child, key=lambda x: x['dist'])
                    actual_alignment = best_cand['pos']
                else:
                    # 見つからなかった場合でも一番近い四角形にフォールバック
                    best_cand = min(candidates, key=lambda x: x['dist'])
                    actual_alignment = best_cand['pos']

        gs, ts = self.grid_size, self.target_side

        # =======================================================
        # デバッグ画像の描画と保存 (debug_mode=True の時のみ)
        # =======================================================
        if debug_mode:
            os.makedirs("sample-result/debug", exist_ok=True)
            base_name = os.path.basename(img_path).replace(".png", "")
            debug_img = img.copy()
            
            # TL(赤), TR(緑), BL(青)
            cv2.circle(debug_img, tuple(map(int, TL)), 3, (0, 0, 255), -1)
            cv2.circle(debug_img, tuple(map(int, TR)), 3, (0, 255, 0), -1)
            cv2.circle(debug_img, tuple(map(int, BL)), 3, (255, 0, 0), -1)
            
            if actual_alignment:
                # アライメント発見(マゼンタ)
                cv2.circle(debug_img, tuple(map(int, actual_alignment)), 3, (255, 0, 255), -1)
                cv2.polylines(debug_img, [np.int32([TL, TR, actual_alignment, BL])], True, (0, 255, 255), 1)
            else:
                # 推測BR(水色)
                cv2.circle(debug_img, tuple(map(int, estimated_BR)), 3, (255, 255, 0), -1)
                cv2.polylines(debug_img, [np.int32([TL, TR, estimated_BR, BL])], True, (0, 255, 255), 1)
                
            cv2.imwrite(f"sample-result/debug/{base_name}_01_anchors.png", debug_img)

        # 透視変換
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
        warped = cv2.warpPerspective(img, M_persp, (ts, ts))

        if debug_mode:
            cv2.imwrite(f"sample-result/debug/{base_name}_02_warped.png", warped)

        return warped

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