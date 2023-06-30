from class_cv import Processing
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# constrói o analisador de argumentos e analisa os argumentos
ap = argparse.ArgumentParser ()
ap. add_argument ( "-i" , "--image" , required = True ,help = "Caminho para a imagem a ser digitalizada" )
args = vars (ap.parse_args ())

# etapa 1 -> detecção de borda
img = cv2.imread(args["image"])
ratio = img.shape[0]/500.0
origem = img.copy()
img = imutils.resize(img, height = 500)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

print('Primeiro passo>>  Detecção de borda')
cv2.imshow("Image", img)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

# etapa 2 ->  encontradando contornos
cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea, reverse=True)[:5]
marcador =  0 
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    print(len(approx))
    if len(approx) == 4:
        screenCnt = approx
        marcador +=1
        break
if  marcador == 0:
    print('Não existe contorno de 4 pontos')
else:
    print("STEP 2: Find contours of paper")
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# etapa 3 > transformação e limiarização
# apply the four point transform to obtain a top-down
# view of the original image
warped = Processing.four_point_transform(origem, screenCnt.reshape(4, 2) * ratio)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imshow("Original", imutils.resize(origem, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey(0)