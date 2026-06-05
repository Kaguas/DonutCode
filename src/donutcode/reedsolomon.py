class _ReedSolomon:
    def __init__(self, prim=0x11d):
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

    def gf_mul(self, x, y):
        if x == 0 or y == 0: return 0
        return self.gf_exp[self.gf_log[x] + self.gf_log[y]]

    def gf_poly_mul(self, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j in range(len(q)):
            for i in range(len(p)):
                r[i + j] ^= self.gf_mul(p[i], q[j])
        return r

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
    
    def decode(self, bit_stream, nsym):
        # ここでは簡略化のため、エラー訂正は実装せず、単純にビットストリームをバイト列に変換して返す
        byte_list = bytearray()
        for i in range(0, len(bit_stream), 8):
            byte_str = bit_stream[i:i+8]
            if len(byte_str) < 8: break
            print("1")
            byte_val = int(byte_str, 2)
            print(f"2: {byte_val}")
            if byte_val == 0:  # 終端(Null)検知
                break
            byte_list.append(byte_val)

        try:
            decoded_text = byte_list.decode('ascii')
            return decoded_text
        except Exception as e:
            print(f"\n デコード失敗 (データ破損): {e}")
            return None