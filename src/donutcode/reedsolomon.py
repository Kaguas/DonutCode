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