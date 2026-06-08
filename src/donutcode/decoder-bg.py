# decoder.py
import cv2
import numpy as np
from .reedsolomon import _ReedSolomon
from .config import get_config
from .vision import VisionProcessor

class Decoder:
    def __init__(self, config_type="D-27-13"):
        self.config = get_config(config_type)
        self.grid_size = self.config.GRID_SIZE
        self.rs = _ReedSolomon()
        self.vision = VisionProcessor(target_side=500)

    def image_to_bitmap(self, square_img):
        """sample.py で成功した正確なサンプリング"""
        gray_crop = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)
        _, thresh_crop = cv2.threshold(gray_crop, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        gs = self.grid_size
        cell_size = square_img.shape[0] / gs
        bit_map = np.zeros((gs, gs), dtype=int)

        for row in range(gs):
            for col in range(gs):
                center_x = int((col + 0.5) * cell_size)
                center_y = int((row + 0.5) * cell_size)
                
                # 黒(<128)なら1、白なら0
                if thresh_crop[center_y, center_x] < 128:  
                    bit_map[row, col] = 1
                else:  
                    bit_map[row, col] = 0

        # ※デジタル画像前提のため、一旦回転補正(_fix_orientation)は省略
        return bit_map

    def _decode_from_bit_map(self, bit_map):
        # 1. 文字数エリアの読み取り (Configを利用)
        char_count = 0
        for (x, y) in self.config.CHAR_COUNT_COORDS:
            char_count = (char_count << 1) | bit_map[y, x]

        # 2. ジグザグマッピングに従って抽出 (Configを利用)
        bit_stream = [bit_map[y, x] for x, y in self.config.get_mapping()]

        # 3. バイト列の復元 (MSB First)
        byte_list = bytearray()
        for i in range(0, len(bit_stream) - (len(bit_stream) % 8), 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | bit_stream[i + j]
            byte_list.append(byte_val)

        # 4. リードソロモンによる誤り訂正と文字列化
        try:
            decoded_bytes = self.rs.decode(byte_list, self.config.ECC_BYTES)
            msg_bytes = bytes(decoded_bytes)
        except Exception as e:
            print(f"[警告] RSデコード失敗: {e}")
            msg_bytes = bytes(byte_list[:-self.config.ECC_BYTES] if len(byte_list) > self.config.ECC_BYTES else byte_list)

        data_bytes = msg_bytes[:char_count] if 0 < char_count <= len(msg_bytes) else msg_bytes
        return data_bytes.rstrip(b'\x00').decode('ascii', errors='ignore')
    
    def decode_image(self, img_path):
        square_img = self.vision.process(img_path)
        bit_map = self.image_to_bitmap(square_img)
        return self._decode_from_bit_map(bit_map)