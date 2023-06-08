import cv2
import numpy as np

img = cv2.imread("unidade 01\decomposicao_img_bits\imgs\desafio-esteganografia.png")

if img is None:
    print("Erro ao abrir a imagem")
    exit()

img_carrier = np.copy(img)
img_encoded = np.copy(img)
nbits = 3

img_carrier = img >> nbits << nbits
img_encoded = img << (8 - nbits)

cv2.imshow("Imagem portadora", img_carrier)
cv2.imshow("Imagem codificada", img_encoded)
cv2.waitKey()