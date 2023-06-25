import cv2
import numpy as np

nbins = 64
range_ = [0, 256]

img = cv2.imread("unidade 01\manipulando_histogramas\imgs\jiji.jpg")

if img is None:
    print("Erro ao abrir a imagem")
    exit()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_eq = cv2.equalizeHist(img)

hist, _ = np.histogram(img, bins=nbins, range=range_)
hist_eq, _ = np.histogram(img_eq, bins=nbins, range=range_)

hist_img = np.zeros((nbins, nbins), dtype=np.uint8)
hist_img_eq = np.zeros((nbins, nbins), dtype=np.uint8)

hist_normalized = np.empty_like(hist, dtype=np.float32)
hist_eq_normalized = np.empty_like(hist_eq, dtype=np.float32)

cv2.normalize(hist, hist_normalized, 0, hist_img.shape[0], cv2.NORM_MINMAX, cv2.CV_32F)
cv2.normalize(hist_eq, hist_eq_normalized, 0, hist_img_eq.shape[0], cv2.NORM_MINMAX, cv2.CV_32F)

hist_img.fill(0)
hist_img_eq.fill(0)

for i in range(nbins):
    cv2.line(hist_img, (i, nbins), (i, nbins - int(np.round(hist_normalized[i]))), (255, 255, 255), 1, 8, 0)
    cv2.line(hist_img_eq, (i, nbins), (i, nbins - int(np.round(hist_eq_normalized[i]))), (255, 255, 255), 1, 8, 0)

img[15:15 + nbins, 15:15 + nbins] = hist_img
img_eq[15:15 + nbins, 15:15 + nbins] = hist_img_eq

cv2.imshow("image", img)
cv2.imshow("equalised_image", img_eq)

cv2.waitKey()
cv2.destroyAllWindows()