import cv2
import numpy as np

def main():
    # Normal img
    img = cv2.imread(r'unidade 01\manipulando_pixels\imgs\totoro.jpg')
    
    if img is None:
        print("Erro ao abrir a imagem")
        exit()

    cv2.imshow('totoro', img)

    rows, cols = img.shape[:2]

    # Getting quadrant img
    quad_1 = img[0:rows//2, 0:cols//2]
    quad_2 = img[0:rows//2, cols//2:cols]
    quad_3 = img[rows//2:rows, 0:cols//2]
    quad_4 = img[rows//2:rows, cols//2:cols]

    new_img = np.empty_like(img)

    # Realocating quadrants in new image
    new_img[0:rows//2, 0:cols//2] = quad_4
    new_img[rows//2:rows, cols//2:cols] = quad_1
    new_img[rows//2:rows, 0:cols//2] = quad_2
    new_img[0:rows//2, cols//2:cols] = quad_3

    cv2.imshow("swapped_totoro", new_img)
    cv2.waitKey()

if __name__ == "__main__":
    main() 