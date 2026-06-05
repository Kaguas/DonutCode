from PIL import Image, ImageDraw
from .reedsolomon import _ReedSolomon

class Encoder:
    def __init__(self, grid_size=21, hole_rect=(7, 7, 7, 7), ecc_bytes=15):
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

    def encode(self, data_str):
        matrix = [[0] * self.grid_size for _ in range(self.grid_size)]
        
        # 基準点を枠外に設定。はみ出したセパレータ部分は _draw_finder 内の範囲チェックで無視される
        self._draw_finder(matrix, -1, -1)                             # 左上
        self._draw_finder(matrix, self.grid_size - 8, -1)             # 右上
        self._draw_finder(matrix, -1, self.grid_size - 8)             # 左下

        hx, hy, hw, hh = self.hole_rect
        for y in range(hy, hy + hh):
            for x in range(hx, hx + hw):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    matrix[y][x] = 2 

        available_cells = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self._is_finder_pattern(x, y): continue
                if self._is_hole(x, y): continue
                available_cells.append((x, y))

        max_bytes = len(available_cells) // 8
        data_bytes_len = max_bytes - self.ecc_bytes

        if data_bytes_len <= 0:
            raise ValueError(f"エラー訂正({self.ecc_bytes}B)に対してデータ領域({max_bytes}B)が小さすぎます。")

        msg_bytes = data_str.encode('ascii', errors='ignore')
        if len(msg_bytes) > data_bytes_len:
            raise ValueError(f"データが長すぎます。最大 {data_bytes_len} バイトまでです。")
        else:
            msg_bytes = msg_bytes + b'\x00' * (data_bytes_len - len(msg_bytes))

        encoded_bytes = self.rs.encode(msg_bytes, self.ecc_bytes)
        bit_stream = []
        for b in encoded_bytes:
            for i in range(7, -1, -1):
                bit_stream.append((b >> i) & 1)
        
        bit_stream += [0] * (len(available_cells) - len(bit_stream))

        for (x, y), bit in zip(available_cells, bit_stream):
            matrix[y][x] = bit

        return matrix

    def save_image(self, matrix, filename, scale=20, hole_color="white",padding=20):
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