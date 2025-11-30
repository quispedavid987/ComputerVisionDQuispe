import cv2
import os
# ---- ARQUITECTURA
prototxt = "models/deploy.prototxt"

# ---- PESOS
model = "models/res10_300x300_ssd_iter_140000.caffemodel"

# --- CARGANDO EL MODELO
net = cv2.dnn.readNetFromCaffe(prototxt, model)

images = os.listdir('./images/')
print(images)

for image in images:
    img = cv2.imread(f'images/{image}')
## LEYENDO UNA IMAGEN

    cv2.imshow("Imagen", img)
    #cv2.imshow("Imagen muestra", image_show)
    #cv2.imshow("Imagen redimensionada", image_resized)
    #cv2.imshow("Imagen pos mean substraction", blob_to_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()