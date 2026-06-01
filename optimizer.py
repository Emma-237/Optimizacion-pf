import numpy as np

class BFGSOptimizer:
    def __init__(self, max_it=15, tolerance=1e-5, rho=0.5):
        self.max_it = max_it
        self.tolerance = tolerance
        self.rho = rho 

    def line_search_biseccion(self, func, xk, pk, color, obj_local, lienzo_local):
        alpha_hat = 0.0
        d_alpha = 1.0 
        epsilon = 1e-4 
        fk = func.eval(xk, color, obj_local, lienzo_local)
        
        while d_alpha > epsilon:
            xk_next = xk + (alpha_hat + d_alpha) * pk
            f_next = func.eval(xk_next, color, obj_local, lienzo_local)
            if f_next < fk:
                alpha_hat = alpha_hat + d_alpha
                fk = f_next
            else:
                d_alpha = self.rho * d_alpha
        return alpha_hat

    def solve(self, func, x0, color, obj_local, lienzo_local):
        xk = np.array(x0, dtype=float).ravel()
        n = len(xk)
        Hk = np.eye(n)
        
        for i in range(self.max_it):
            gk = func.diff(xk, color, obj_local, lienzo_local)
            if np.linalg.norm(gk) < self.tolerance:
                break
            pk = -(Hk @ gk)
            if np.dot(gk, pk) > 0:
                Hk = np.eye(n)
                pk = -gk
            alpha = self.line_search_biseccion(func, xk, pk, color, obj_local, lienzo_local)
            sk = alpha * pk
            xk_next = xk + sk
            yk = func.diff(xk_next, color, obj_local, lienzo_local) - gk
            ys = np.dot(yk, sk)
            if abs(ys) > 1e-12:
                rho_k = 1.0 / ys
                Hk = (np.eye(n) - rho_k * np.outer(sk, yk)) @ Hk @ (np.eye(n) - rho_k * np.outer(yk, sk)) + (rho_k * np.outer(sk, sk))
            xk = xk_next
            
            xk[2] = np.clip(xk[2], 1.5, 12.0)
            xk[3] = np.clip(xk[3], 1.5, 12.0)
            xk[5] = np.clip(xk[5], 0.1, 0.45)
        return xk