import cv2
import numpy as np
import random

STEP = 5
JITTER = 3
RAIO = 3

top_slider = 10
top_slider_max = 200

def pointillism(image, border, points):
    height, width = image.shape[:2]
    x_indices = list(range(0, height, STEP))
    y_indices = list(range(0, width, STEP))
    random.shuffle(x_indices)
    random.shuffle(y_indices)

    for i in x_indices:
        for j in y_indices:
            if border[i, j] == 255:
                x = i + random.randint(-JITTER, JITTER + 1)
                y = j + random.randint(-JITTER, JITTER + 1)
                color = image[x, y]
                cv2.circle(points, (y, x), RAIO, (int(color[0]), int(color[1]), int(color[2])), -1, cv2.LINE_AA)
            else:
                x = i + random.randint(-JITTER, JITTER + 1)
                y = j + random.randint(-JITTER, JITTER + 1)
                color = image[x, y]
                cv2.circle(points, (y, x), 2, (int(color[0]), int(color[1]), int(color[2])), -1, cv2.LINE_AA)

    cv2.imshow("cannypoints", points)

def on_trackbar_canny(value):
    global border
    _, thresholded = cv2.threshold(imageGray, value, 255, cv2.THRESH_BINARY)
    border = cv2.Canny(thresholded, 30, 90)
    pointillism(image, border, points)
    cv2.imshow("canny", border)

def on_trackbar_cannypoints(value):
    pointillism(image, border, points)

image = cv2.imread(r'unidade 02\canny_pontilhismo\img\ponyo.jpg', cv2.IMREAD_COLOR)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
points = image.copy()
border = np.zeros(imageGray.shape, dtype=np.uint8)

if image is None:
    print("Erro ao abrir a imagem")
    exit()

cv2.namedWindow("cannypoints", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold inferior", "cannypoints", top_slider, top_slider_max, on_trackbar_canny)
on_trackbar_canny(top_slider)

cv2.namedWindow("trackbars", cv2.WINDOW_NORMAL)

cv2.createTrackbar("STEP", "trackbars", STEP, 100, on_trackbar_cannypoints)
on_trackbar_cannypoints(STEP)

cv2.createTrackbar("JITTER", "trackbars", JITTER, 100, on_trackbar_cannypoints)
on_trackbar_cannypoints(JITTER)

cv2.createTrackbar("RAIO", "trackbars", RAIO, 100, on_trackbar_cannypoints)
on_trackbar_cannypoints(RAIO)

while True:
    if cv2.waitKey(1) == 27:
        cv2.imwrite("edges_detected.png", border)
        cv2.imwrite("ponyo_pointillism.png", points)
        break
