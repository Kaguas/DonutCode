import cv2
import numpy as np
import os

class VisionProcessor:
    def __init__(self, config=None, target_side=500):
        self.config = config
        self.grid_size = config.GRID_SIZE if config else 27
        self.target_side = target_side

    def _get_finder_candidates(self, contours, hierarchy, strict_mode=True):
        centers, valid_contours = [], []
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

    def _find_alignment(self, contours, hierarchy, estimated_BR, search_radius, TL, TR, BL):
        """アライメントパターンを探す：内側に子要素(黒四角)を持つ白四角を最優先"""
        candidates = []
        for i in range(len(contours)):
            c = contours[i]
            # 既にファインダとして検出済みの輪郭を除外するため面積や距離でフィルタ
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            
            # 推定位置付近であること
            if np.hypot(cx - estimated_BR[0], cy - estimated_BR[1]) < search_radius:
                # ファインダ自身のノードではない（かつ距離的に遠い）ことを保証
                if (np.hypot(cx - TL[0], cy - TL[1]) > search_radius and
                    np.hypot(cx - TR[0], cy - TR[1]) > search_radius and
                    np.hypot(cx - BL[0], cy - BL[1]) > search_radius):
                    
                    # 階層構造: 内側に黒四角(子要素)があるか確認 (hierarchy[i][2] != -1)
                    has_child = hierarchy[0][i][2] != -1
                    candidates.append({'pos': [cx, cy], 'has_child': has_child})

        if not candidates: return None
        # 内側黒四角を持つものを優先
        with_child = [c for c in candidates if c['has_child']]
        return with_child[0]['pos'] if with_child else candidates[0]['pos']

    def process(self, img_path, debug_mode=False):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # ファインダ検出 (堅牢な2Dアプローチのみに一本化)
        centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=True)
        if len(centers) < 3:
            centers, valid_contours = self._get_finder_candidates(contours, hierarchy, strict_mode=False)
        
        if len(centers) < 3: return self._fallback_crop(img, thresh)

        # 幾何学計算
        pts = np.array(centers[:3], dtype=np.float32)
        dists = [np.linalg.norm(pts[0]-pts[1]), np.linalg.norm(pts[1]-pts[2]), np.linalg.norm(pts[0]-pts[2])]
        max_d = max(dists)
        if dists[0] == max_d: tl, tr, bl = 2, 0, 1
        elif dists[1] == max_d: tl, tr, bl = 0, 1, 2
        else: tl, tr, bl = 1, 0, 2
        
        TL, TR, BL = pts[tl], pts[tr], pts[bl]
        if np.cross(TR - TL, BL - TL) > 0: TR, BL = BL, TR
        estimated_BR = TR + BL - TL

        # アライメント検出
        actual_alignment = None
        if self.config and hasattr(self.config, 'ALIGNMENT_POS'):
            cell_size = (np.linalg.norm(TR - TL) + np.linalg.norm(BL - TL)) / (2 * (self.grid_size - 7))
            actual_alignment = self._find_alignment(contours, hierarchy, estimated_BR, cell_size * 4, TL, TR, BL)

        # 透視変換
        ts, gs = self.target_side, self.grid_size
        src = np.float32([TL, TR, BL, actual_alignment if actual_alignment else estimated_BR])
        dst = np.float32([
            [3.5*(ts/gs), 3.5*(ts/gs)], [(gs-3.5)*(ts/gs), 3.5*(ts/gs)],
            [3.5*(ts/gs), (gs-3.5)*(ts/gs)], 
            ([self.config.ALIGNMENT_POS[0]+2.5, self.config.ALIGNMENT_POS[1]+2.5] * (ts/gs)) if actual_alignment else [(gs-3.5)*(ts/gs), (gs-3.5)*(ts/gs)]
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (ts, ts))

        if debug_mode:
            os.makedirs("sample-result/debug", exist_ok=True)
            dbg = img.copy()
            for p in [TL, TR, BL]: cv2.circle(dbg, tuple(map(int, p)), 5, (0,0,255), -1)
            if actual_alignment: cv2.circle(dbg, tuple(map(int, actual_alignment)), 5, (255,0,255), -1)
            cv2.imwrite(f"sample-result/debug/{os.path.basename(img_path)}_debug.png", dbg)

        return warped

    def _fallback_crop(self, img, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return img
        rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        box = cv2.boxPoints(rect)
        ts = self.target_side
        dst = np.float32([[0,0], [ts,0], [ts,ts], [0,ts]])
        M = cv2.getPerspectiveTransform(np.float32(box), dst)
        return cv2.warpPerspective(img, M, (ts, ts))