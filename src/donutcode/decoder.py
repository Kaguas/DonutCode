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

    def _fix_orientation(self, bit_map):
        """
        4隅の7x7領域を、理想的なファインダーパターンのテンプレートと比較し、
        最も一致しない角を「ファインダーの無いデータ領域（右下）」と判定する。
        """
        gs = self.grid_size
        
        # 7x7の理想的なファインダーパターン（1:黒, 0:白）
        # これがどんな角度から見ても同じ形である性質を利用します
        fp_template = np.array([
            [1,1,1,1,1,1,1],
            [1,0,0,0,0,0,1],
            [1,0,1,1,1,0,1],
            [1,0,1,1,1,0,1],
            [1,0,1,1,1,0,1],
            [1,0,0,0,0,0,1],
            [1,1,1,1,1,1,1]
        ])

        # 4隅の7x7領域を正確に抽出
        corners = {
            'TL': bit_map[0:7, 0:7],
            'TR': bit_map[0:7, gs-7:gs],
            'BL': bit_map[gs-7:gs, 0:7],
            'BR': bit_map[gs-7:gs, gs-7:gs]
        }

        # テンプレートとの一致ピクセル数（最大49）を計算
        match_scores = {}
        for name, corner_img in corners.items():
            match_scores[name] = np.sum(corner_img == fp_template)

        print(f" -> [回転判定] パターン一致度: TL={match_scores['TL']}/49, TR={match_scores['TR']}/49, BL={match_scores['BL']}/49, BR={match_scores['BR']}/49")

        # 最も一致度が低い（スコアが最小の）角が、ファインダーがない角
        odd_corner = min(match_scores, key=match_scores.get)
        print(f" -> [回転判定] ファインダーの無い角は '{odd_corner}' と判定されました。")

        if odd_corner == 'BR':
            print(" -> [回転補正] 正しい向きのため回転しません。")
        elif odd_corner == 'TR':
            bit_map = np.rot90(bit_map, -1)  # 時計回りに90度回転
            print(" -> [回転補正] 時計回りに90度回転させました。")
        elif odd_corner == 'TL':
            bit_map = np.rot90(bit_map, 2)   # 180度回転
            print(" -> [回転補正] 180度回転させました。")
        elif odd_corner == 'BL':
            bit_map = np.rot90(bit_map, 1)   # 反時計回りに90度回転
            print(" -> [回転補正] 反時計回りに90度回転させました。")

        return bit_map

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

        # サンプリング完了後に回転補正をかける
        return self._fix_orientation(bit_map)

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

        stripped_byte_list = bytearray(byte_list)
        while len(stripped_byte_list) > self.ecc_bytes and stripped_byte_list[-1] == 0:
            stripped_byte_list.pop()

        try:
            decoded = self.rs.decode(stripped_byte_list, self.ecc_bytes)
            msg_bytes = decoded[0] if isinstance(decoded, tuple) else decoded
            msg_bytes = bytes(msg_bytes).rstrip(b'\x00')
            return msg_bytes.decode('utf-8', errors='ignore')
            
        except Exception as e:
            raw_bytes = bytearray()
            for b in byte_list:
                if b == 0: break
                raw_bytes.append(b)
            try:
                return raw_bytes.decode('utf-8', errors='ignore')
            except:
                return None 

    def decode_image(self, img_path):
        src_img = cv2.imread(img_path)
        if src_img is None:
            raise FileNotFoundError(f"画像 '{img_path}' が見つかりません。")

        gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        largest_contour = max(contours, key=cv2.contourArea)
        pts = largest_contour.reshape(-1, 2)
        
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        box = np.zeros((4, 2), dtype="float32")
        box[0] = pts[np.argmin(s)]
        box[1] = pts[np.argmin(diff)]
        box[2] = pts[np.argmax(s)]
        box[3] = pts[np.argmax(diff)]

        square_img = self.warp_to_square(src_img, box, side_length=500)
        bit_map = self.image_to_bitmap(square_img)
        
        decoded_text = self._decode_from_bit_map(bit_map)
        
        if decoded_text:
            return [decoded_text]
        
        return []