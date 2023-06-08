import cv2
import numpy as np

nbins = 64
range_ = [0, 256]

cap = cv2.VideoCapture("unidade 01\manipulando_histogramas\imgs\jiji.mp4")
if not cap.isOpened():
    print("Erro abrindo o video!")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    ret, img = cap.read()
    if not ret:
        break

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

    cv2.imshow("video", img)
    cv2.imshow("equalised_video", img_eq)
    if cv2.waitKey(30) >= 0:
        break

cap.release()
cv2.destroyAllWindows()