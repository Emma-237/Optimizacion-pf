import numpy as np
import cv2

class PincelFunction:
    def eval(self, params, color, obj_local, lienzo_local):
        m = self._generar_mascara(obj_local.shape, params)
        proyeccion = lienzo_local * (1 - m) + color * m
        return np.mean((obj_local - proyeccion)**2)

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
        # Simplificación de pincelada elíptica
        dist = ((x - p[0])**2 / (2 * p[2]**2) + (y - p[1])**2 / (2 * p[3]**2))
        mask = np.exp(-dist) * p[5] # p[5] es la opacidad
        return mask[:, :, np.newaxis]

def ejecutar_agente(imagen_objetivo, iteraciones=2000):
    h, w, _ = imagen_objetivo.shape
    lienzo = np.ones((h, w, 3))
    opt = BFGSOptimizer()
    func = PincelFunction()

    for i in range(iteraciones):
        x0, y0 = np.random.randint(0, w), np.random.randint(0, h)
        color_target = imagen_objetivo[y0, x0, :]
        
        # Parámetros iniciales: x, y, sigma_x, sigma_y, theta, alpha
        p_inicial = np.array([x0, y0, 20.0, 20.0, 0.0, 0.5])
        
        # Extraer región local para optimizar rápido
        p_optimo = opt.solve(func, p_inicial, color_target, imagen_objetivo, lienzo)
        
        # Aplicar pincelada final al lienzo
        mask = func._generar_mascara(lienzo.shape, p_optimo)
        lienzo = lienzo * (1 - mask) + color_target * mask
        
        if i % 100 == 0:
            print(f"Pincelada {i} completada...")
            cv2.imshow("Progreso", lienzo)
            cv2.waitKey(1)

    return lienzo