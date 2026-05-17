import cv2
import numpy as np
import os
from optimizer import BFGSOptimizer
from funciones import PincelFunction

class RegionManager:
    def __init__(self, ancho, alto, tam_cuadro=80):
        self.ancho, self.alto = ancho, alto
        self.tam_cuadro = tam_cuadro
        self.cuadros = [(y, x) for y in range(0, alto, tam_cuadro) for x in range(0, ancho, tam_cuadro)]
        self.cuadro_actual = 0

    def obtener_siguiente_cuadro(self):
        if self.cuadro_actual >= len(self.cuadros): return None
        y, x = self.cuadros[self.cuadro_actual]
        self.cuadro_actual += 1
        return (y, min(y + self.tam_cuadro, self.alto), x, min(x + self.tam_cuadro, self.ancho))

def cargar_imagen(ruta):
    img = cv2.imread(ruta)
    if img is None: exit(f"Error: No se encontró {ruta}")
    w_f = 600
    h_f = int(img.shape[0] * (w_f / img.shape[1]))
    img = cv2.resize(img, (w_f, h_f))
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0

def pintar():
    carpeta = "pinceladas_final"
    if not os.path.exists(carpeta): os.makedirs(carpeta)

    target = cargar_imagen("gato.jpg")
    h, w, _ = target.shape
    
    # Lógica de reanudación basada en tus archivos previos[cite: 2]
    archivos = [f for f in os.listdir(carpeta) if f.endswith(".png")]
    if archivos:
        numeros = [int(f.split('_')[1].split('.')[0]) for f in archivos]
        ultimo = max(numeros)
        lienzo = cv2.cvtColor(cv2.imread(f"{carpeta}/cuadro_{ultimo:03d}.png"), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inicio = ultimo
    else:
        lienzo, inicio = np.zeros_like(target), 0

    manager = RegionManager(w, h)
    manager.cuadro_actual = inicio
    func, opt = PincelFunction(), BFGSOptimizer(max_it=100) # Usando BFGSOptimizer[cite: 3]

    while True:
        reg = manager.obtener_siguiente_cuadro()
        if reg is None: break
        y1, y2, x1, x2 = reg
        obj_l, lien_l = target[y1:y2, x1:x2], lienzo[y1:y2, x1:x2]
        
        for _ in range(500): 
            lx, ly = np.random.randint(0, obj_l.shape[1]), np.random.randint(0, obj_l.shape[0])
            color = np.mean(obj_l[max(0, ly-1):ly+2, max(0, lx-1):lx+2], axis=(0, 1))
            
            if np.mean(color) < 0.08: 
                # Pincelada de limpieza
                p_opt, c_u = np.array([float(lx), float(ly), 2.0, 2.0, 0.0, 0.4]), np.zeros(3)
            else: 
                # Parámetros iniciales optimizados para evitar manchas[cite: 2, 3]
                p0 = [lx, ly, 4.0, 4.0, 0.0, 0.2]
                p_opt = opt.solve(func, p0, color, obj_l, lien_l)
                c_u = color

            p_g = p_opt.copy()
            p_g[0] += x1; p_g[1] += y1
            
            # --- AQUÍ ESTABA EL ERROR: Cambiado a _generar_mascara ---
            mask = func._generar_mascara(lienzo.shape, p_g)
            lienzo = lienzo * (1 - mask) + c_u * mask

        # Guardado en formato BGR para OpenCV[cite: 2]
        img_bgr = cv2.cvtColor((lienzo*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{carpeta}/cuadro_{manager.cuadro_actual:03d}.png", img_bgr)
        print(f"Cuadro {manager.cuadro_actual} listo.")

if __name__ == "__main__":
    pintar()