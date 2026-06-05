import cv2
import numpy as np
from .reedsolomon import _ReedSolomon

class Decoder:
    def __init__(self, grid_size=21, hole_rect=(7, 7, 7, 7), ecc_bytes=24):
        self.grid_size = grid_size
        self.hole_rect = hole_rect
        self.ecc_bytes = ecc_bytes
        self.rs = _ReedSolomon()

    def _is_finder_pattern(self, x, y):
        if 0 <= x < 8 and 0 <= y < 8: return True
        if self.grid_size - 8 <= x < self.grid_size and 0 <= y < 8: return True
        if 0 <= x < 8 and self.grid_size - 8 <= y < self.grid_size: return True
        return False

    def _is_hole(self, x, y):
        hx, hy, hw, hh = self.hole_rect
        return hx <= x < hx + hw and hy <= y < hy + hh

    @staticmethod
    def _order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def warp_to_square(self, src_img, box, side_length=500):
        pts1 = self._order_points(box.astype("float32"))
        pts2 = np.float32([
            [0, 0], 
            [side_length, 0], 
            [side_length, side_length], 
            [0, side_length]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(src_img, M, (side_length, side_length))

    def image_to_bitmap(self, square_img):
        if len(square_img.shape) == 3:
            gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = square_img
            
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        side_length = square_img.shape[0]
        cell_size = side_length / self.grid_size
        bit_map = np.zeros((self.grid_size, self.grid_size), dtype=int)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                center_x = int((col + 0.5) * cell_size)
                center_y = int((row + 0.5) * cell_size)

                if thresh[center_y, center_x] < 128:
                    bit_map[row, col] = 1
                else:
                    bit_map[row, col] = 0

        return bit_map

    def _decode_from_bit_map(self, bit_map):
        available_cells = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_finder_pattern(x, y): continue
                if self._is_hole(x, y): continue
                available_cells.append((x, y))

        bit_stream = [bit_map[y, x] for x, y in available_cells]

        byte_list = bytearray()
        for i in range(0, len(bit_stream) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bit_stream[i + j]
            byte_list.append(byte_val)

        # 【追加】エンコーダーが余白を埋めたゼロパディングを取り除く
        stripped_byte_list = bytearray(byte_list)
        while len(stripped_byte_list) > self.ecc_bytes and stripped_byte_list[-1] == 0:
            stripped_byte_list.pop()

        try:
            # 正規のReed-Solomonエラー訂正ルート
            decoded = self.rs.decode(stripped_byte_list, self.ecc_bytes)
            msg_bytes = decoded[0] if isinstance(decoded, tuple) else decoded
            msg_bytes = bytes(msg_bytes).rstrip(b'\x00')
            return msg_bytes.decode('utf-8', errors='ignore')
            
        except Exception as e:
            print(f"⚠️ Reed-Solomon エラー訂正に失敗しました: {e}")
            print(" -> [フォールバック] 生データの直接抽出を試みます...")
            
            # 【追加】エラー訂正なしのフォールバックルート（以前成功したロジック）
            raw_bytes = bytearray()
            for b in byte_list:
                if b == 0: break  # Null(終端)検知でストップ
                raw_bytes.append(b)
                
            try:
                return raw_bytes.decode('utf-8', errors='ignore')
            except Exception as e2:
                print(f"❌ フォールバック抽出にも失敗しました: {e2}")
                return None 

    def decode_image(self, img_path):
        src_img = cv2.imread(img_path)
        if src_img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が見つかりません。")

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        coords = cv2.findNonZero(thresh)
        if coords is None:
            return []

        rect = cv2.minAreaRect(coords)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        square_img = self.warp_to_square(src_img, box, side_length=500)
        bit_map = self.image_to_bitmap(square_img)
        
        decoded_text = self._decode_from_bit_map(bit_map)
        
        if decoded_text:
            return [decoded_text]
        
        return []