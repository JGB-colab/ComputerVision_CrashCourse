# 1 :  Detectar o exame em uma imagem
# 2 :  Aplicar a transformação de perspectiva(visão panorâmica)
# 3 :  Extrair o conjunto de bolhas
# 4 :  Classificar as perguntas/balões em linhas
# 5 :  Determinar as respostas marcadas
# 6 :  Validar as respostas
# 7 :  Repitir para todos os exames

from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

# Processando a imagem
img = cv2.imread(r'day4\Teste_bolha.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(blur,75,200)

# Pespectiva do documento
cnts = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
doccnt = None

if len(cnts)>0:
    cnts = sorted(cnts,key= cv2.contourArea,reverse = True)
    for c in cnts:
        peri = cv2.arcLength(c,True)
        aprox = cv2.approxPolyDP(c,0.02*peri,True)
        if len(aprox) == 4:
            doccnt = aprox
            break
# Aplicando 
paper = four_point_transform(img,doccnt.reshape(4,2))
warped = four_point_transform(gray,doccnt.reshape(4,2))

'''cv2.imshow('paper',paper)
cv2.imshow('warped',warped)
cv2.waitKey(0)'''

thresh = cv2.threshold(warped,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Extraindo contornos

cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionsCnts = []

for c in cnts:
    (x,y,w,h) = cv2.boundingRect(c)
    ar = w/float(h)
    if w>=20 and ar>= 0.9 and ar <1.1:
        questionsCnts.append(c)

# Passo de detecção linhas de contornos
questionsCnts = contours.sort_contours(questionsCnts,method='top-to-bottom')[0]
correct = 0

for (q,i) in enumerate(np.arange(0,len(questionsCnts),5)):
    cnts = contours.sort_contours(questionsCnts[i:i + 5])[0]
    bubbled = None


# Identificar caixa selecionada

    for (j,c) in enumerate(cnts):
        mask = np.zeros(thresh.shape,dtype='uint8')
        cv2.drawContours(mask,[c],-1,255,-1)

        mask = cv2.bitwise_and(thresh,thresh,mask = mask)
        total = cv2.countNonZero(mask)
        if bubbled is None or total > bubbled[0]:
            bubbled= (total,j)
        
        color = (0,0,255)
        k = ANSWER_KEY[q]

        if k == bubbled[1]:
            color = (0,255,0)
            correct +=1

        cv2.drawContours(paper,[cnts[k]],-1,color,3)

# grab the test taker
score = (correct / 5.0) * 100 - 200
print("[INFO] score: {:.2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
	cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", img)
cv2.imshow("Exam", paper)
cv2.waitKey(0)