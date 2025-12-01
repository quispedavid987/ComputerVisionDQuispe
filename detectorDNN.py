import cv2
import os
import time
import albumentations as A #---- AGREGANDO ALBUMENTATIONS PARA ROTACIONES
# ---- ARQUITECTURA
prototxt = "models/deploy.prototxt"

# ---- PESOS
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

# --- CARGANDO EL MODELO
net = cv2.dnn.readNetFromCaffe(prototxt, model)

images = sorted(os.listdir('./images/'))

# ROTAMOS, CAMBIAMOS EL BRILLO Y SIMULAMOS OCLUSION
transform = A.Compose([A.SafeRotate(limit=30,p=1.0),A.RandomBrightnessContrast(p=1.0),A.CoarseDropout(num_holes_range=(1,2),hole_height_range=(10,20),hole_width_range=(10,20),p=1.0),])

time_cumulated = 0
confidence = 0.90
print(f"Grado de confianza en detección : {confidence}")
print(f"Imagen\tTiempo( ms)\tRostros")
for img in images:
	start = time.time() ## CUANTIFICANDO TIEMPO PARA PROCESO DE CADA IMAGEN (INICIO)
	## LEYENDO UNA IMAGEN
	image = cv2.imread(f"images/{img}")

	# ---- APLICAMOS AUGMENTATION
	## CONVERTIMOS A RGB
	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# ---- APLICAMOS LA TRANSFORMACION
	augmented = transform(image=image_rgb)["image"]

	# ---- REGRESAMOS AL FORMATO PARA EL MODELO DNNS
	image_def = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)

	## REDIMENSIONANDO PARA QUE SEA IMAGEN FUNCIONAL CON LA RED
	height, width, _ = image_def.shape
	image_resized = cv2.resize(image_def, (300,300))

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
			cv2.rectangle(image_def, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
			cv2.putText(image_def, f"{detection[2]*100:.1f}", (x_start, y_start-5), 1, 1.2, (0,0,255), 2)
			detected_faces += 1

	end = time.time()
	time_cumulated += end-start
	print(f'{img}\t{(end-start)*1000:.2f}\t{detected_faces}\t')
	## MOSTRANDO IMAGEN

	cv2.imwrite(f'results_aug/{img}_cfd_0{confidence*100}.jpg',image_def)
	#cv2.imshow("Imagen", image)
	#cv2.imshow("Imagen muestra", image_show)
	#cv2.imshow("Imagen redimensionada", image_resized)
	#cv2.imshow("Imagen pos mean substraction", blob_to_show)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

print(f'Tiempo promedio por imagen : {time_cumulated/len(images)*1000:.2f} milisegundos')