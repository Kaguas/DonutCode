# encoder.py
import colorsys
from PIL import Image, ImageDraw
from .reedsolomon import _ReedSolomon
from .config import get_config

class Encoder:
    def __init__(self, config_type="D-27-13"):
        self.config = get_config(config_type)
        self.grid_size = self.config.GRID_SIZE
        self.rs = _ReedSolomon()

    def _draw_finder(self, matrix, ox, oy):
        """
        全体で9x9の領域を処理します。
        一番外側の1マスをセパレータ（白）、内側7x7をファインダパタン（黒枠・白枠・黒中心）とします。
        配列の範囲外（はみ出した部分）は無視されるため、3つの角すべてに同じ関数を適用できます。
        """ 
        for dy in range(9):
            for dx in range(9):
                x = ox + dx
                y = oy + dy
                
                # 【重要】Pythonは -1 を指定すると末尾を書き換えてしまうため、必ず範囲チェックを行う
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    # 1層目(最外周): セパレータ (白:0)
                    if dx == 0 or dx == 8 or dy == 0 or dy == 8:
                        matrix[y][x] = 0
                    # 2層目: ファインダパタン外枠 (黒:1)
                    elif dx == 1 or dx == 7 or dy == 1 or dy == 7:
                        matrix[y][x] = 1
                    # 3層目: ファインダパタン内枠 (白:0)
                    elif dx == 2 or dx == 6 or dy == 2 or dy == 6:
                        matrix[y][x] = 0
                    # 4層目(中心の3x3): (黒:1)
                    else:
                        matrix[y][x] = 1

    def _draw_fixed_patterns(self, matrix):
        """Configの判定メソッドと連携して固定パターンを描画"""
        
        # 1. ファインダパタンの描画 (基準点をはみ出させて9x9を共通描画)
        self._draw_finder(matrix, -1, -1)                              # 左上
        self._draw_finder(matrix, self.grid_size - 8, -1)              # 右上
        self._draw_finder(matrix, -1, self.grid_size - 8)              # 左下

        # 2. その他のパターンの描画
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.config.is_finder(x, y):
                    # ファインダはすでに描画済みのためスキップ
                    continue
                
                elif self.config.is_alignment(x, y):
                    ax, ay = self.config.ALIGNMENT_POS
                    dx, dy = x - ax, y - ay
                    # 5x5のアライメントパターン
                    if dx == 0 or dx == 4 or dy == 0 or dy == 4: matrix[y][x] = 1
                    elif dx == 2 and dy == 2: matrix[y][x] = 1
                    else: matrix[y][x] = 0

                elif self.config.is_timing(x, y):
                    # ゼブラ模様のタイミングパターン
                    matrix[y][x] = 1 if (x if y == 7 else y) % 2 == 0 else 0

    def _draw_character_count(self, matrix, count):
        """Configの座標順序に従ってMSBから書き込む"""
        bits = [(count >> i) & 1 for i in range(7, -1, -1)]
        for (x, y), bit in zip(self.config.CHAR_COUNT_COORDS, bits):
            matrix[y][x] = bit
    
    def encode(self, data_str):
        matrix = [[0] * self.grid_size for _ in range(self.grid_size)]
        
        # 固定パターンの描画
        self._draw_fixed_patterns(matrix)

        available_cells = self.config.get_mapping()
        max_bytes = len(available_cells) // 8
        data_bytes_len = max_bytes - self.config.ECC_BYTES
        
        if data_bytes_len <= 0:
            raise ValueError("データ領域が小さすぎます。")

        msg_bytes = data_str.encode('ascii', errors='ignore')
        if len(msg_bytes) > data_bytes_len:
            raise ValueError("データが長すぎます。")

        self._draw_character_count(matrix, len(msg_bytes))
        
        # QR風パディング
        padding = bytes([0xEC if i % 2 == 0 else 0x11 for i in range(data_bytes_len - len(msg_bytes))])
        full_msg_bytes = msg_bytes + padding

        # RSエンコード
        encoded_bytes = self.rs.encode(full_msg_bytes, self.config.ECC_BYTES)

        # ビット化 (MSB First)
        bit_stream = [(byte >> i) & 1 for byte in encoded_bytes for i in range(7, -1, -1)]
        bit_stream.extend([0] * (len(available_cells) - len(bit_stream)))

        # Configのマッピング順序で配置
        for (x, y), bit in zip(available_cells, bit_stream):
            matrix[y][x] = bit

        return matrix

    def save_mapping_debug_image(self, filename, scale=20, padding=20):
        img = Image.new("RGB", (self.grid_size * scale + 2 * padding, self.grid_size * scale + 2 * padding), "white")
        draw = ImageDraw.Draw(img)
        
        available_cells = self.config.get_mapping()
        total_bytes = len(available_cells) // 8
        
        cell_to_byte_idx = {}
        for bit_idx, (x, y) in enumerate(available_cells):
            byte_idx = bit_idx // 8
            cell_to_byte_idx[(x, y)] = byte_idx

        def get_gradient_color(idx, total):
            if total <= 0: return (128, 128, 128)
            hue = idx / total
            r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            return (int(r * 255), int(g * 255), int(b * 255))

        hx, hy, hw, hh = self.config.HOLE_RECT

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                box = [x * scale + padding, y * scale + padding, (x + 1) * scale + padding, (y + 1) * scale + padding]
                
                if (x, y) in cell_to_byte_idx:
                    b_idx = cell_to_byte_idx[(x, y)]
                    if b_idx < total_bytes:
                        color = get_gradient_color(b_idx, total_bytes)
                    else:
                        color = (200, 200, 200)
                    draw.rectangle(box, fill=color, outline="white")
                
                elif self.config.is_finder(x, y) or self.config.is_alignment(x, y) or self.config.is_timing(x, y):
                    draw.rectangle(box, fill="black")
                
                elif self.config.is_char_count(x, y):
                    draw.rectangle(box, fill="#FFD700", outline="white")
                
                elif hx <= x < hx + hw and hy <= y < hy + hh:
                    draw.rectangle(box, fill="#FFE4E1")
                
                else:
                    draw.rectangle(box, fill="#F0F0F0", outline="white")
                    
        img.save(filename)

    def save_image(self, matrix, filename, scale=20, hole_color="white", padding=20):
        size = self.grid_size
        img = Image.new("RGB", (size * scale + 2 * padding, size * scale + 2 * padding), "white")
        draw = ImageDraw.Draw(img)
        for y in range(size):
            for x in range(size):
                val = matrix[y][x]
                box = [x * scale + padding, y * scale + padding, (x + 1) * scale + padding, (y + 1) * scale + padding]
                if val == 1:
                    draw.rectangle(box, fill="black")
                elif val == 0:
                    draw.rectangle(box, fill="white")
                elif val == 2:
                    draw.rectangle(box, fill=hole_color) 
        img.save(filename)