import cv2

img = cv2.imread("unidade 01\decomposicao_img_bits\imgs\desafio-esteganografia.png")

if img is None:
    print("Erro ao abrir a imagem")
    exit()

rows, cols = img.shape[:2]
print(img[0])

cv2.imshow("img", img)
cv2.waitKey()