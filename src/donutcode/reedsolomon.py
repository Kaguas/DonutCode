# reedsolomon.py

class _ReedSolomon:
    def __init__(self, prim=0x11d):
        # ガロア体 GF(2^8) の指数・対数テーブルの生成
        self.gf_exp = [1] * 512
        self.gf_log = [0] * 256
        x = 1
        for i in range(1, 255):
            x <<= 1
            if x & 0x100:
                x ^= prim
            self.gf_exp[i] = x
            self.gf_log[x] = i
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]

    # --- ガロア体上の基本演算 ---
    def gf_mul(self, x, y):
        if x == 0 or y == 0: return 0
        return self.gf_exp[self.gf_log[x] + self.gf_log[y]]

    def gf_div(self, x, y):
        if y == 0: raise ZeroDivisionError("GF(2^8) Division by zero")
        if x == 0: return 0
        return self.gf_exp[(self.gf_log[x] + 255 - self.gf_log[y]) % 255]

    def gf_inv(self, x):
        return self.gf_exp[255 - self.gf_log[x]]

    def gf_poly_scale(self, p, x):
        return [self.gf_mul(p[i], x) for i in range(len(p))]

    def gf_poly_add(self, p, q):
        r = [0] * max(len(p), len(q))
        for i in range(len(p)): r[i + len(r) - len(p)] = p[i]
        for i in range(len(q)): r[i + len(r) - len(q)] ^= q[i]
        return r

    def gf_poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self.gf_mul(p[i], q[j])
        return r

    def gf_poly_eval(self, p, x):
        y = p[0]
        for i in range(1, len(p)):
            y = self.gf_mul(y, x) ^ p[i]
        return y

    # --- エンコード（書き込み） ---
    def rs_generator_poly(self, nsym):
        g = [1]
        for i in range(nsym):
            g = self.gf_poly_mul(g, [1, self.gf_exp[i]])
        return g

    def encode(self, msg_bytes, nsym):
        gen = self.rs_generator_poly(nsym)
        msg_out = list(msg_bytes) + [0] * nsym
        for i in range(len(msg_bytes)):
            coef = msg_out[i]
            if coef != 0:
                for j in range(1, len(gen)):
                    msg_out[i + j] ^= self.gf_mul(gen[j], coef)
        msg_out[:len(msg_bytes)] = list(msg_bytes)
        return bytes(msg_out)

    # --- デコード（誤り訂正）のための各アルゴリズム ---
    def rs_calc_syndromes(self, msg, nsym):
        synd = [0] * nsym
        for i in range(nsym):
            synd[i] = self.gf_poly_eval(msg, self.gf_exp[i])
        return [0] + synd # インデックス調整のためのパディング

    def rs_find_error_locator(self, synd, nsym):
        """バーレカンプ・マッシー法でエラー位置多項式を計算"""
        err_loc = [1]
        old_loc = [1]
        for i in range(nsym):
            K = i + 1
            delta = synd[K]
            for j in range(1, len(err_loc)):
                delta ^= self.gf_mul(err_loc[-(j+1)], synd[K - j])
            
            old_loc = old_loc + [0]
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self.gf_poly_scale(old_loc, delta)
                    old_loc = self.gf_poly_scale(err_loc, self.gf_inv(delta))
                    err_loc = new_loc
                err_loc = self.gf_poly_add(err_loc, self.gf_poly_scale(old_loc, delta))
        return err_loc

    def rs_find_errors(self, err_loc, nmess):
        """チェン探索でエラーの位置を特定"""
        errs = len(err_loc) - 1
        err_pos = []
        for i in range(nmess):
            if self.gf_poly_eval(err_loc, self.gf_exp[255 - i]) == 0:
                err_pos.append(nmess - 1 - i)
        if len(err_pos) != errs:
            raise ValueError("Too many errors to correct")
        return err_pos

    def rs_find_error_evaluator(self, synd, err_loc, nsym):
        remainder = self.gf_poly_mul(synd, err_loc)
        return remainder[-(nsym+1):]

    def rs_correct_errata(self, msg_in, synd, err_pos):
        """フォーニー・アルゴリズムでエラーの大きさを計算して修復"""
        msg = list(msg_in)
        coef_pos = [len(msg) - 1 - p for p in err_pos]
        err_loc = self.rs_find_error_locator(synd, len(synd) - 1)
        err_eval = self.rs_find_error_evaluator(synd, err_loc, len(synd) - 1)

        X = []
        for i in range(len(coef_pos)):
            X.append(self.gf_exp[255 - coef_pos[i]])

        for i, Xi in enumerate(X):
            Xi_inv = self.gf_inv(Xi)
            
            err_loc_prime = 1
            for j in range(len(X)):
                if j != i:
                    err_loc_prime = self.gf_mul(err_loc_prime, 1 ^ self.gf_mul(Xi_inv, X[j]))
            
            y = self.gf_poly_eval(err_eval, Xi_inv)
            y = self.gf_mul(X[i], y)
            
            magnitude = self.gf_div(y, err_loc_prime)
            msg[err_pos[i]] ^= magnitude # ビット反転(XOR)による修復
            
        return msg

    # --- 外部から呼び出されるメイン関数 ---
    def decode(self, msg_in, nsym):
        """
        受信したバイト列（実データ＋パリティ）からエラーを検出し、
        可能であれば訂正した実データ部分のバイト列を返します。
        """
        msg = list(msg_in)
        
        # 1. シンドローム計算（エラー検知）
        synd = self.rs_calc_syndromes(msg, nsym)
        
        # すべてのシンドロームが0なら、エラーなし（そのまま実データ部を返す）
        if max(synd) == 0:
            return bytes(msg[:-nsym])
            
        # 2. エラーがある場合は訂正を試みる
        try:
            err_loc = self.rs_find_error_locator(synd, nsym)
            err_pos = self.rs_find_errors(err_loc, len(msg))
            msg_corrected = self.rs_correct_errata(msg, synd, err_pos)
            
            print(f" -> [Reed-Solomon] {len(err_pos)} バイトのデータ破損を検知し、自動修復しました！")
            return bytes(msg_corrected[:-nsym])
            
        except ValueError:
            # 修復限界を超えている場合
            raise ValueError(f"データが修復限界を超えて破損しています（パリティ {nsym} バイトでは修復不可）")