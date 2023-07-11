import cv2
import numpy as np

nClusters = 8
nRodadas = 1

img = cv2.imread(r'unidade 02\kmeans\img\sushi.jpg', cv2.IMREAD_COLOR)

if img is None:
    print("Erro ao abrir imagem")
    exit()

for i in range(10):
    samples = img.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
    _, labels, centers = cv2.kmeans(samples, nClusters, None, criteria, nRodadas, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape(img.shape[:2])
    rotulada = np.zeros_like(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            indice = labels[y, x]
            rotulada[y, x] = centers[indice]

    cv2.imwrite("sushi_clusterizado_{}.png".format(i), rotulada)
