import cv2
import numpy as np
import os
import glob
from optimizer import BFGSOptimizer
from funciones import PincelFunction

def obtener_ultimo_archivo(carpeta):
    archivos = glob.glob(os.path.join(carpeta, "detalle_*.png"))
    if not archivos:
        return None, 0
    ultimo = max(archivos, key=os.path.getctime)
    try:
        iteracion = int(ultimo.split('_')[-1].split('.')[0])
    except:
        iteracion = 0
    return ultimo, iteracion

def refinar():
    carpeta_refinados = "refinados"
    if not os.path.exists(carpeta_refinados):
        os.makedirs(carpeta_refinados)

    ultimo_refinado, inicio_it = obtener_ultimo_archivo(carpeta_refinados)
    
    # 1. Cargar la mejor base disponible[cite: 2, 4]
    if ultimo_refinado:
        print(f"Reanudando desde: {ultimo_refinado}")
        ruta_base = ultimo_refinado
    else:
        ruta_base = "pinceladas/cuadro_042.png" 
        print(f"Iniciando desde base original: {ruta_base}")

    img_previa = cv2.imread(ruta_base)
    if img_previa is None:
        print(f"Error fatal: No se encontró la imagen en {ruta_base}")
        return

    lienzo = cv2.cvtColor(img_previa.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
    
    # 2. Cargar objetivo y ajustar tamaño[cite: 2]
    target_img = cv2.imread("gato.jpg")
    target_img = cv2.resize(target_img, (lienzo.shape[1], lienzo.shape[0]))
    target = cv2.cvtColor(target_img.astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
    
    h, w, _ = target.shape
    func = PincelFunction()
    opt = BFGSOptimizer(max_it=10) # Pocas iteraciones para velocidad[cite: 3]

    print("Refinando con parches locales (Velocidad Turbo)...")

    # 3. Bucle de refinamiento con parches
    tam = 20 # Radio del parche (cuadro de 40x40)
    
    for i in range(inicio_it, 20001): 
        # Seleccionar punto aleatorio evitando bordes extremos
        x = np.random.randint(tam, w - tam)
        y = np.random.randint(tam, h - tam)
        
        color_target = target[y, x, :]
        
        # EXTRACCIÓN DEL PARCHE: Solo procesamos esta zona
        obj_local = target[y-tam:y+tam, x-tam:x+tam]
        lienzo_local = lienzo[y-tam:y+tam, x-tam:x+tam]
        
        # p0 relativo al centro del parche (tam, tam)
        p0 = np.array([float(tam), float(tam), 3.0, 3.0, 0.0, 0.3])
        
        # Optimización local (Mucho más rápida)[cite: 3]
        p_opt = opt.solve(func, p0, color_target, obj_local, lienzo_local)
        
        # Restringir a pinceles muy finos para detalles[cite: 3]
        p_opt[2] = np.clip(p_opt[2], 1, 6) 
        p_opt[3] = np.clip(p_opt[3], 1, 6)
        
        # Ajustar coordenadas al lienzo global
        p_global = p_opt.copy()
        p_global[0] += (x - tam)
        p_global[1] += (y - tam)
        
        # Aplicar pincelada al lienzo completo[cite: 1, 4]
        mask = func._generar_mascara(lienzo.shape, p_global)
        lienzo = lienzo * (1 - mask) + color_target * mask

        # Guardar cada 100 para no perder tiempo en escritura de disco
        if i % 100 == 0 and i != inicio_it:
            nombre = f"{carpeta_refinados}/detalle_{i:05d}.png"
            img_save = cv2.cvtColor((lienzo * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(nombre, img_save)
            print(f"Pincelada {i} terminada.")

if __name__ == "__main__":
    refinar()