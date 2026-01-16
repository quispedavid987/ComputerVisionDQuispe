import numpy as np
import cv2
import os
import time

def tanhSimple(image, bbox):
	'''
	Version simplificada
	'''
	x, y, w, h = bbox
	height, width = image.shape[:2]
	
	## COORDENADAS
	cx, cy = x + w//2, y + h//2
	a = w/(2*np.sqrt(np.pi))
	b = h/(2*np.sqrt(np.pi))
	
	## MAPA DE REMAPEO
	map_x = np.zeros((height, width), dtype = np.float32)
	map_y = np.zeros((height, width), dtype = np.float32)
	
	## MAPEO DE PIXELES
	yi, xi = np.indices((height, width))
	di, dj = yi - cy, xi -cx
	
	## ECUACIONES
	rho = np.tanh(np.sqrt(di**2 + dj**2)/(np.sqrt(a**2 + b**2)+ 1e-6))
	theta = np.tanh(dj/(di + 1e-6))
	
	## PROYECCION PARA DETECCION SECUNDARIA
	map_x = cx + theta*(w/2)
	map_y = cy + rho*(h/2)
	
	return cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)
	

# ---- ARQUITECTURA
prototxt = "models/deploy.prototxt"

# ---- PESOS
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

# --- CARGANDO EL MODELO
net = cv2.dnn.readNetFromCaffe(prototxt, model)

images = sorted(os.listdir('./images/'))

time_cumulated = 0
confidence = 0.30

print(f"Grado de confianza en detección : {confidence}")
print(f"Imagen\tTiempo( ms)\tRostros")

for img in images:
	start = time.time() ## CUANTIFICANDO TIEMPO PARA PROCESO DE CADA IMAGEN (INICIO)
	## LEYENDO UNA IMAGEN
	image = cv2.imread(f"images/{img}")

	## REDIMENSIONANDO PARA QUE SEA IMAGEN FUNCIONAL CON LA RED
	height, width, _ = image.shape
	image_resized = cv2.resize(image, (300,300))

	## PREPROCESAMIENTO
	blob = cv2.dnn.blobFromImage(image_resized, 1, (300,300), (104, 117, 123)) # PARAMETROS STANDAR PARA LA RED
	#print(img, " blob shape : ", blob.shape)
	blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

	detected_faces = 0
	## DETECCION
	net.setInput(blob) ## PREPARANDO EL BLOB COMO ENTRADA DE LA RED
	detections = net.forward()
	
	k = 0
	for detection in detections[0][0]:
		if detection[2] > confidence: ## CAMBIANDO EL VALOR DE CONFIANZA
			box = detection[3:7]*[width, height, width, height] ## SE EXTRAE LAS COORDENADAS DE LA DETECCION
			x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			w, h = x_end - x_start, y_end - y_start
			## APLICANDO LA TRANSFORMACION TANH SIMPLE
			warped = tanhSimple(image, (x_start, y_start, w, h))
			
			cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
			cv2.putText(image, f"{detection[2]*100:.1f}", (x_start, y_start-5), 1, 1.2, (0,0,255), 2)
			cv2.imwrite(f'results_1/{img}_warped_{k}.jpg',warped)
			detected_faces += 1
			k += 1

	end = time.time()
	time_cumulated += end-start
	print(f'{img}\t{(end-start)*1000:.2f}\t{detected_faces}\t')
	## MOSTRANDO IMAGEN

	cv2.imwrite(f'results_1/{img}_cfd_0{confidence*100}.jpg',image)
	cv2.imshow("Imagen", image)
	#cv2.imshow("Imagen muestra", image)
	#cv2.imshow("Imagen redimensionada", image_resized)
	#cv2.imshow("Imagen pos mean substraction", blob_to_show)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

print(f'Tiempo promedio por imagen : {time_cumulated/len(images)*1000:.2f} milisegundos')

