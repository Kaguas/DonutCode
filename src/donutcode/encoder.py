from PIL import Image, ImageDraw
from .reedsolomon import _ReedSolomon

class Encoder:
    def __init__(self, grid_size=21, hole_rect=(7, 7, 7, 7), ecc_bytes=15):
        self.grid_size = grid_size
        self.hole_rect = hole_rect
        self.ecc_bytes = ecc_bytes
        self.rs = _ReedSolomon()

    def _is_finder_pattern(self, x, y):
        if 0 <= x < 7 and 0 <= y < 7: return True
        if self.grid_size - 7 <= x < self.grid_size and 0 <= y < 7: return True
        if 0 <= x < 7 and self.grid_size - 7 <= y < self.grid_size: return True
        return False

    def _is_hole(self, x, y):
        hx, hy, hw, hh = self.hole_rect
        return hx <= x < hx + hw and hy <= y < hy + hh

    def _draw_finder(self, matrix, ox, oy):
        for dy in range(7):
            for dx in range(7):
                if dx == 0 or dx == 6 or dy == 0 or dy == 6:
                    matrix[oy + dy][ox + dx] = 1
                elif dx == 1 or dx == 5 or dy == 1 or dy == 5:
                    matrix[oy + dy][ox + dx] = 0
                else:
                    matrix[oy + dy][ox + dx] = 1

    def encode(self, data_str):
        matrix = [[0] * self.grid_size for _ in range(self.grid_size)]
        self._draw_finder(matrix, 0, 0)
        self._draw_finder(matrix, self.grid_size - 7, 0)
        self._draw_finder(matrix, 0, self.grid_size - 7)

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

        msg_bytes = data_str.encode('utf-8', errors='ignore')
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

    def save_image(self, matrix, filename, scale=20, hole_color="white"):
        size = self.grid_size
        img = Image.new("RGB", (size * scale, size * scale), "white")
        draw = ImageDraw.Draw(img)
        for y in range(size):
            for x in range(size):
                val = matrix[y][x]
                box = [x * scale, y * scale, (x + 1) * scale, (y + 1) * scale]
                if val == 1:
                    draw.rectangle(box, fill="black")
                elif val == 0:
                    draw.rectangle(box, fill="white")
                elif val == 2:
                    draw.rectangle(box, fill=hole_color) 
        img.save(filename)