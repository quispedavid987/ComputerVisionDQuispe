import cv2
import os
import time
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
	for detection in detections[0][0]:
		if detection[2] > confidence: ## CAMBIANDO EL VALOR DE CONFIANZA
			box = detection[3:7]*[width, height, width, height] ## SE EXTRAE LAS COORDENADAS DE LA DETECCION
			x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
			cv2.putText(image, f"{detection[2]*100:.1f}", (x_start, y_start-5), 1, 1.2, (0,0,255), 2)
			detected_faces += 1

	end = time.time()
	time_cumulated += end-start
	print(f'{img}\t{(end-start)*1000:.2f}\t{detected_faces}\t')
	## MOSTRANDO IMAGEN

	cv2.imshow("Imagen", image)
	#cv2.imshow("Imagen muestra", image_show)
	#cv2.imshow("Imagen redimensionada", image_resized)
	#cv2.imshow("Imagen pos mean substraction", blob_to_show)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

print(f'Tiempo promedio por imagen : {time_cumulated/len(images)*1000:.2f} milisegundos')