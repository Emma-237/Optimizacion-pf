import cv2
import numpy as np
import os
from optimizer import BFGSOptimizer
from funciones import PincelFunction

class RegionManager:
    def __init__(self, ancho, alto, tam_cuadro=150):
        self.ancho = ancho
        self.alto = alto
        self.tam_cuadro = tam_cuadro
        self.cuadros = []
        for y in range(0, alto, tam_cuadro):
            for x in range(0, ancho, tam_cuadro):
                self.cuadros.append((y, x))
        self.cuadro_actual = 0

    def obtener_siguiente_cuadro(self):
        if self.cuadro_actual >= len(self.cuadros):
            return None
        y, x = self.cuadros[self.cuadro_actual]
        y2 = min(y + self.tam_cuadro, self.alto)
        x2 = min(x + self.tam_cuadro, self.ancho)
        self.cuadro_actual += 1
        return (y, y2, x, x2)

def cargar_imagen(ruta):
    img = cv2.imread(ruta)
    if img is None:
        print(f"Error: No se encontró {ruta}")
        exit()
    ancho_f = 600
    alto_f = int(img.shape[0] * (ancho_f / img.shape[1]))
    img = cv2.resize(img, (ancho_f, alto_f))
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0

def pintar():
    carpeta_salida = "pinceladas_blanco_fino4"
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Recuerda renombrar tu archivo a 'gato.jpg' o cambiar este texto
    target = cargar_imagen("gato.jpg")
    h, w, _ = target.shape
    
    archivos = [f for f in os.listdir(carpeta_salida) if f.endswith(".png")]
    if archivos:
        numeros = [int(f.split('_')[1].split('.')[0]) for f in archivos]
        ultimo_cuadro = max(numeros)
        nombre_ultimo = f"{carpeta_salida}/cuadro_{ultimo_cuadro:03d}.png"
        
        print(f"Continuando desde progreso guardado: {nombre_ultimo}")
        lienzo_bgr = cv2.imread(nombre_ultimo)
        lienzo = cv2.cvtColor(lienzo_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inicio_cuadro = ultimo_cuadro
    else:
        print("Iniciando pintura con fondo blanco impecable...")
        lienzo = np.ones_like(target) 
        inicio_cuadro = 0

    manager = RegionManager(w, h, tam_cuadro=150) 
    manager.cuadro_actual = inicio_cuadro

    func = PincelFunction()
    opt = BFGSOptimizer(max_it=15, tolerance=1e-4) 

    print(f"Procesando cuadros a partir del {inicio_cuadro + 1}...")

    while True:
        region = manager.obtener_siguiente_cuadro()
        if region is None: 
            print("¡Proceso completado con alta calidad!")
            break
        
        y1, y2, x1, x2 = region
        obj_local = target[y1:y2, x1:x2]
        
        # 1200 pinceladas por bloque para un pelaje denso y rápido
        for p in range(2500): 
            lx = np.random.randint(0, obj_local.shape[1])
            ly = np.random.randint(0, obj_local.shape[0])
            
            y_in, y_fi = max(0, ly-1), min(obj_local.shape[0], ly+2)
            x_in, x_fi = max(0, lx-1), min(obj_local.shape[1], lx+2)
            color = np.mean(obj_local[y_in:y_fi, x_in:x_fi], axis=(0, 1)) 
            
            # Si el fondo original ya es blanco, no gasta cómputo y lo deja blanco.
            if np.all(color > 0.96):
                continue
            
            lienzo_local = lienzo[y1:y2, x1:x2]
            p0 = np.array([float(lx), float(ly), 3.0, 3.0, 0.0, 0.25])
            p_opt = opt.solve(func, p0, color, obj_local, lienzo_local)
            
            p_global = p_opt.copy()
            p_global[0] += x1
            p_global[1] += y1
            
            mask = func._generar_mascara(lienzo.shape, p_global)
            lienzo = lienzo * (1 - mask) + color * mask

        nombre = f"{carpeta_salida}/cuadro_{manager.cuadro_actual:03d}.png"
        img_bgr = cv2.cvtColor((lienzo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(nombre, img_bgr)
        print(f"Cuadro {manager.cuadro_actual} guardado de forma rápida.")

if __name__ == "__main__":
    pintar()