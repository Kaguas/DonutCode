import cv2
import numpy as np

class Decoder:
    def __init__(self, grid_size=21, hole_rect=(7, 7, 7, 7)):
        self.grid_size = grid_size
        self.hole_rect = hole_rect

    def _is_finder_pattern(self, x, y):
        if 0 <= x < 7 and 0 <= y < 7: return True
        if self.grid_size - 7 <= x < self.grid_size and 0 <= y < 7: return True
        if 0 <= x < 7 and self.grid_size - 7 <= y < self.grid_size: return True
        return False

    def _is_hole(self, x, y):
        hx, hy, hw, hh = self.hole_rect
        return hx <= x < hx + hw and hy <= y < hy + hh

    def _decode_from_bit_map(self, bit_map):
        bit_stream = ""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self._is_finder_pattern(col, row): continue
                if self._is_hole(col, row): continue
                bit_stream += str(bit_map[row, col])

        byte_list = bytearray()
        for i in range(0, len(bit_stream), 8):
            byte_str = bit_stream[i:i+8]
            if len(byte_str) < 8:
                break
            byte_val = int(byte_str, 2)
            if byte_val == 0: # 0x00 (Null) でデータの終端を検知
                break
            byte_list.append(byte_val)

        try:
            return byte_list.decode('utf-8')
        except Exception:
            return None 

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _extract_bit_map(self, cropped_img):
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # 大津の2値化で白黒を明確に分ける
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_h, img_w = thresh.shape
        cell_h = img_h / self.grid_size
        cell_w = img_w / self.grid_size

        bit_map = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                center_x = int((col + 0.5) * cell_w)
                center_y = int((row + 0.5) * cell_h)
                
                # 【修正点1】エンコーダーに合わせて「黒い部分（128未満）」を 1 とする
                if thresh[center_y, center_x] < 128:
                    bit_map[row, col] = 1 
                else:
                    bit_map[row, col] = 0 
        return bit_map

    def decode_image(self, img_path):
        src_img = cv2.imread(img_path)
        if src_img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が見つかりません。")

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        
        # 画像内の黒いピクセルを白(255)として抽出
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 【修正点2】コードの枠線を「すべての黒ピクセルの座標の最小矩形」から計算
        # 3隅に四角いファインダーパターンがあるため、この矩形はピッタリとコード全体を囲みます。
        coords = cv2.findNonZero(thresh)
        if coords is None:
            return []

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        pts1 = self._order_points(box.astype("float32"))
        side_length = 300
        pts2 = np.float32([[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]])

        # パースペクティブ変換で正面化
        M = cv2.getPerspectiveTransform(pts1, pts2)
        cropped_img = cv2.warpPerspective(src_img, M, (side_length, side_length))

        # ビットマップ化してデコード
        bit_map = self._extract_bit_map(cropped_img)
        decoded_text = self._decode_from_bit_map(bit_map)
        
        if decoded_text:
            return [decoded_text]
        
        return []