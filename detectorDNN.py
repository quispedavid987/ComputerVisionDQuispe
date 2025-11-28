import cv2
# ---- ARQUITECTURA
prototxt = "models/deploy.prototxt"

# ---- PESOS
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

# --- CARGANDO EL MODELO
net = cv2.dnn.readNetFromCaffe(prototxt, model)

## LEYENDO UNA IMAGEN
image = cv2.imread("images/imagen_7.jpg")

## REDIMENSIONANDO PARA QUE SEA IMAGEN FUNCIONAL CON LA RED
height, width, _ = image.shape
image_resized = cv2.resize(image, (300,300))

## PREPROCESAMIENTO
blob = cv2.dnn.blobFromImage(image_resized, 1, (300,300), (104, 117, 123))
print(blob.shape)
blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

## DETECCION
net.setInput(blob) ## ENTRADA DE LA RED
detections = net.forward()
for detection in detections[0][0]:
	if detection[2] > 0.5: ## CAMBIANDO EL VALOR DE CONFIANZA
		box = detection[3:7]*[width, height, width, height]
		x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
		cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0,0,255), 2)
		cv2.putText(image, f"{detection[2]*100:.1f}", (x_start, y_start-5), 1, 1.2, (0,0,255), 2)
## MOSTRANDO IMAGEN

cv2.imshow("Imagen", image)
#cv2.imshow("Imagen muestra", image_show)
#cv2.imshow("Imagen redimensionada", image_resized)
#cv2.imshow("Imagen pos mean substraction", blob_to_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
