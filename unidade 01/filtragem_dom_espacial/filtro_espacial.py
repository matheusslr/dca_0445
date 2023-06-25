import cv2
import numpy as np

def printmask(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print(m[i, j], end=",")
        print()

cap = cv2.VideoCapture("unidade 01\manipulando_histogramas\imgs\jiji.jpg")  # open the default camera
if not cap.isOpened():
    print("[ERROR]: Could not open the video")
    exit(1)

media = np.array([[0.1111, 0.1111, 0.1111],
                  [0.1111, 0.1111, 0.1111],
                  [0.1111, 0.1111, 0.1111]], dtype=np.float32)
gauss = np.array([[0.0625, 0.125, 0.0625],
                  [0.125, 0.25, 0.125],
                  [0.0625, 0.125, 0.0625]], dtype=np.float32)
horizontal = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
vertical = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)
laplacian = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]], dtype=np.float32)
boost = np.array([[0, -1, 0],
                  [-1, 5.2, -1],
                  [0, -1, 0]], dtype=np.float32)

_, frame = cap.read()
height, width, _ = frame.shape
print("largura =", width)
print("altura =", height)
print("fps =", cap.get(cv2.CAP_PROP_FPS))
print("formato =", cap.get(cv2.CAP_PROP_FORMAT))

cv2.namedWindow("filtroespacial", cv2.WINDOW_NORMAL)
cv2.namedWindow("original", cv2.WINDOW_NORMAL)

mask = np.zeros((3, 3), dtype=np.float32)
mask[:, :] = media

absolut = 1  # calcs abs of the image

while True:
    # _, frame = cap.read()  # get a new frame from camera
    framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    framegray = cv2.flip(framegray, 1)
    cv2.imshow("original", framegray)
    frame32f = framegray.astype(np.float32)
    frameFiltered = cv2.filter2D(frame32f, -1, mask, anchor=(1, 1), delta=0)

    if absolut:
        frameFiltered = np.abs(frameFiltered)

    result = frameFiltered.astype(np.uint8)

    cv2.imshow("filtroespacial", result)

    key = cv2.waitKey(30)
    if key == 27:  # esc pressed!
        break
    elif key == ord('a'):
        absolut = not absolut
    elif key == ord('m'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = media
        printmask(mask)
    elif key == ord('g'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = gauss
        printmask(mask)
    elif key == ord('h'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = horizontal
        printmask(mask)
    elif key == ord('v'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = vertical
        printmask(mask)
    elif key == ord('l'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = laplacian
        printmask(mask)
    elif key == ord('b'):
        mask = np.zeros((3, 3), dtype=np.float32)
        mask[:, :] = boost

cap.release()
cv2.destroyAllWindows()