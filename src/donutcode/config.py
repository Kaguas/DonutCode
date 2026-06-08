# config.py
class Config_D_27_13:
    GRID_SIZE = 27
    HOLE_RECT = (7, 7, 13, 13)
    ECC_BYTES = 24  # データエリアが小さいため、ECCは多めに取る

    # アライメントパターンの左上座標
    ALIGNMENT_POS = (20, 20)

    # 文字数エリア(8ビット)の座標リスト (MSB -> LSBの順)
    CHAR_COUNT_COORDS = [(x, y) for y in range(8, 10) for x in range(4)]

    @classmethod
    def is_finder(cls, x, y):
        gs = cls.GRID_SIZE
        if 0 <= x < 8 and 0 <= y < 8: return True
        if gs - 8 <= x < gs and 0 <= y < 8: return True
        if 0 <= x < 8 and gs - 8 <= y < gs: return True
        return False

    @classmethod
    def is_hole(cls, x, y):
        hx, hy, hw, hh = cls.HOLE_RECT
        return hx <= x < hx + hw and hy <= y < hy + hh

    @classmethod
    def is_alignment(cls, x, y):
        ax, ay = cls.ALIGNMENT_POS
        return ax <= x < ax + 5 and ay <= y < ay + 5
    
    @classmethod
    def is_timing(cls, x, y):
        if y == 7 and 8 <= x <= cls.GRID_SIZE - 9: return True
        if x == 7 and 8 <= y <= cls.GRID_SIZE - 9: return True
        return False

    @classmethod
    def is_char_count(cls, x, y):
        return (x, y) in cls.CHAR_COUNT_COORDS

    @classmethod
    def get_mapping(cls):
        """右下から2列ずつ左へ進むジグザグスキャンの座標リストを取得"""
        available = []
        gs = cls.GRID_SIZE
        upward = True  
        
        for x_base in range(gs - 1, -1, -2):
            x_coords = [x_base, x_base - 1] if x_base > 0 else [x_base]
            y_range = range(gs - 1, -1, -1) if upward else range(gs)
            
            for y in y_range:
                for x in x_coords:
                    # 予約領域はスキップ
                    if cls.is_finder(x, y): continue
                    if cls.is_hole(x, y): continue
                    if cls.is_alignment(x, y): continue
                    if cls.is_char_count(x, y): continue
                    if cls.is_timing(x, y): continue
                    
                    available.append((x, y))
            upward = not upward
            
        return available

CONFIG_REGISTRY = {
    "D-27-13": Config_D_27_13,
}

def get_config(type_name):
    if type_name not in CONFIG_REGISTRY:
        raise ValueError(f"未定義のコンフィグタイプです: {type_name}")
    return CONFIG_REGISTRY[type_name]