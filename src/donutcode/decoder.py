import cv2
import numpy as np
from .reedsolomon import _ReedSolomon # エンコーダーに合わせてインポートを追加

class Decoder:
    def __init__(self, grid_size=21, hole_rect=(7, 7, 7, 7), ecc_bytes=15): # ecc_bytesを追加
        self.grid_size = grid_size
        self.hole_rect = hole_rect
        self.ecc_bytes = ecc_bytes
        self.rs = _ReedSolomon() # Reed-Solomonを初期化

    def _is_finder_pattern(self, x, y):
        if 0 <= x < 8 and 0 <= y < 8: return True
        if self.grid_size - 8 <= x < self.grid_size and 0 <= y < 8: return True
        if 0 <= x < 8 and self.grid_size - 8 <= y < self.grid_size: return True
        return False

    def _is_hole(self, x, y):
        hx, hy, hw, hh = self.hole_rect
        return hx <= x < hx + hw and hy <= y < hy + hh

    def _decode_from_bit_map(self, bit_map):
        # エンコーダーと全く同じ順序で利用可能セルを取得
        available_cells = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_finder_pattern(x, y): continue
                if self._is_hole(x, y): continue
                available_cells.append((x, y))

        # ビットストリームの再構築
        bit_stream = [bit_map[y, x] for x, y in available_cells]

        print(f"Bit stream: {bit_stream}") # デバッグ用にビットストリームを表示

        # ビットをバイト配列に変換
        byte_list = bytearray()
        for i in range(0, len(bit_stream) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bit_stream[i + j]
            byte_list.append(byte_val)

        try:
            # Reed-Solomonによるエラー訂正
            decoded = self.rs.decode(byte_list, self.ecc_bytes)
            print(f"Decoded: {decoded}") # デバッグ用にデコード結果を表示
            # 実装によって rs.decode がタプル (メッセージ, ECC) を返すか、単なるバイト列を返すかに両対応
            msg_bytes = decoded[0] if isinstance(decoded, tuple) else decoded
            
            # エンコーダーでデータサイズ調整のために付与した 0x00 (Null) パディングを右側から除去
            msg_bytes = msg_bytes.rstrip(b'\x00')
            
            return msg_bytes.decode('ascii')
        except Exception as e:
            # エラー訂正の限界を超えている、またはノイズが多すぎる場合
            print(f"Decode error: {e}") # 必要に応じてデバッグ出力
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
                
                # 黒い部分（128未満）を 1 とする
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

        # 全ての黒ピクセルの座標の最小矩形を取得
        # (エンコーダーの改修により、ファインダパタンの外枠黒線が必ず端に来るためこのロジックが活きます)
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