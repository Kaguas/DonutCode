# vision.py
import cv2
import numpy as np

class VisionProcessor:
    def __init__(self, grid_size=27, target_side=500):
        self.grid_size = grid_size
        self.target_side = target_side
        self.cell_size = target_side / grid_size

    def process(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が読み込めません。")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 画像サイズから、ファインダがすっぽり入るブロックサイズを動的計算
        min_dim = min(gray.shape[0], gray.shape[1])
        block_size = int((min_dim / self.grid_size) * 8)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        block_size = max(31, min(block_size, 255)) # 31〜255の奇数に制限

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 10
        )

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

        BR = TR + BL - TL

        c = self.cell_size
        gs = self.grid_size
        
        src_pts = np.float32([TL, TR, BL, BR])
        dst_pts = np.float32([
            [(3.5) * c, (3.5) * c],
            [(gs - 3.5) * c, (3.5) * c],
            [(3.5) * c, (gs - 3.5) * c],
            [(gs - 3.5) * c, (gs - 3.5) * c]
        ])

        M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M_persp, (self.target_side, self.target_side))

        return warped

    def _fallback_crop(self, img, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return img
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
        
        dst_pts = np.float32([[0, 0], [self.target_side, 0], [self.target_side, self.target_side], [0, self.target_side]])
        M = cv2.getPerspectiveTransform(rect_pts, dst_pts)
        return cv2.warpPerspective(img, M, (self.target_side, self.target_side))