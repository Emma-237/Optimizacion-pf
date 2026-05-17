import numpy as np

class PincelFunction:
    def eval(self, params, color, obj_local, lienzo_local):
        m = self._generar_mascara(obj_local.shape, params)
        proyeccion = lienzo_local * (1 - m) + color * m
        mse = np.mean((obj_local - proyeccion)**2)
        
        # Penalización: El error aumenta si el pincel crece demasiado (params[2] y [3])
        # Esto obliga al optimizador a mantenerse en rangos de detalle.
        reg = 0.0005 * (params[2]**2 + params[3]**2)
        return mse + reg

    def diff(self, params, color, obj_local, lienzo_local):
        epsilon = 1e-3
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += epsilon
            p_minus = params.copy()
            p_minus[i] -= epsilon
            grad[i] = (self.eval(p_plus, color, obj_local, lienzo_local) - 
                       self.eval(p_minus, color, obj_local, lienzo_local)) / (2 * epsilon)
        return grad

    def _generar_mascara(self, shape, p):
        y, x = np.ogrid[:shape[0], :shape[1]]
        cos_t, sin_t = np.cos(p[4]), np.sin(p[4])
        x_rot = (x - p[0]) * cos_t + (y - p[1]) * sin_t
        y_rot = -(x - p[0]) * sin_t + (y - p[1]) * cos_t
        # p[2] y p[3] son los radios sigma
        dist = (x_rot**2 / (2 * p[2]**2 + 1e-6) + y_rot**2 / (2 * p[3]**2 + 1e-6))
        mask = np.exp(-dist) * p[5] # p[5] es opacidad
        return mask[:, :, np.newaxis]