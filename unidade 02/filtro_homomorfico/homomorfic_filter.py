import cv2
import numpy as np
import math

gh, gl, c, d0 = 1.0, 0.5, 1.0, 1.0
gh_slider, gl_slider, c_slider, d0_slider = 1, 1, 1, 10
gh_max, gl_max, c_max, d0_max = 200, 100, 100, 200

def swapQuadrants(imagem):
    qtd_colunas = imagem.shape[1]
    qtd_linhas = imagem.shape[0]
    centerX = qtd_colunas // 2
    centerY = qtd_linhas // 2
    imagem_modificada = np.empty_like(imagem)
    imagem_modificada[:centerY, :centerX] = imagem[centerY:, centerX:]
    imagem_modificada[:centerY, centerX:] = imagem[centerY:, :centerX]
    imagem_modificada[centerY:, :centerX] = imagem[:centerY, centerX:]
    imagem_modificada[centerY:, centerX:] = imagem[:centerY, :centerX]
    return imagem_modificada

def filtro(gl, gh, c, d0, padded):
    dft_M, dft_N = padded.shape[0], padded.shape[1]

    dist_x = cv2.distanceTransform(np.arange(dft_N, dtype=np.uint8).reshape(1, -1), cv2.DIST_L2, 0)
    dist_y = cv2.distanceTransform(np.arange(dft_M, dtype=np.uint8).reshape(-1, 1), cv2.DIST_L2, 0)
    dist_squared = (dist_x ** 2 + dist_y ** 2) / (d0 ** 2)
    exponent = -c * dist_squared
    filter2D = (gh - gl) * (1 - np.exp(exponent)) + gl

    cv2.imshow("filtro", filter2D)

    filter2D = cv2.normalize(filter2D, None, 0, 1, cv2.NORM_MINMAX)
    filter = np.dstack([filter2D.astype(np.float32), np.zeros_like(filter2D).astype(np.float32)])

    return filter

def aplicar_filtro():
    global gh, gl, c, d0
    dft_M = cv2.getOptimalDFTSize(image.shape[0])
    dft_N = cv2.getOptimalDFTSize(image.shape[1])
    padded = cv2.copyMakeBorder(image, 0, dft_M - image.shape[0], 0, dft_N - image.shape[1], cv2.BORDER_CONSTANT, value=0)

    planos = [np.float32(padded), np.zeros_like(padded, dtype=np.float32)]
    complexImage = cv2.merge(planos)

    complexImage = cv2.dft(complexImage)
    complexImage = swapQuadrants(complexImage)
    filter = filtro(gl, gh, c, d0, padded)

    complexImage = cv2.mulSpectrums(complexImage, filter, 0)

    complexImage = swapQuadrants(complexImage)
    complexImage = cv2.idft(complexImage)

    planos = cv2.split(complexImage)
    result = planos[0]

    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)

    return result


def on_trackbar_gh(value):
    global gh
    gh = value/100.0

def on_trackbar_gl(value):
    global gl
    gl = value /100.0

def on_trackbar_c(value):
    global c
    c = value/10.0

def on_trackbar_d0(value):
    global d0
    d0 = value

image = cv2.imread(r'unidade 02\filtro_homomorfico\img\local_escuro.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("original", image)
if image is None:
    print("Erro abrindo imagem")
    exit(1)

cv2.namedWindow("img_final")
cv2.namedWindow("trackbars")

TrackbarName = "gh - {}".format(gh_max)
cv2.createTrackbar(TrackbarName, "trackbars", gh_slider, gh_max, on_trackbar_gh)

TrackbarName = "gl - {}".format(gl_max)
cv2.createTrackbar(TrackbarName, "trackbars", gl_slider, gl_max, on_trackbar_gl)

TrackbarName = "c - {}".format(c_max)
cv2.createTrackbar(TrackbarName, "trackbars", c_slider, c_max, on_trackbar_c)

TrackbarName = "d0 - {}".format(d0_max)
cv2.createTrackbar(TrackbarName, "trackbars", d0_slider, d0_max, on_trackbar_d0)

while True:
    imagem_final = aplicar_filtro()

    cv2.imshow("img_final", imagem_final)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
