import cv2
import numpy as np

def deslocaDFT(image):
    cx = image.shape[1] // 2
    cy = image.shape[0] // 2
    image[:cy, :cx], image[cy:, cx:] = image[cy:, cx:].copy(), image[:cy, :cx].copy()
    image[cy:, :cx], image[:cy, cx:] = image[:cy, cx:].copy(), image[cy:, :cx].copy()

def on_trackbar():
    pass

def main():
    image = cv2.imread(r'unidade 02\filtro_homomorfico\img\ceu_escuro.jpg', cv2.IMREAD_GRAYSCALE)

    # Calcula os tamanhos ótimos para o cálculo da DFT
    dft_M = cv2.getOptimalDFTSize(image.shape[0])
    dft_N = cv2.getOptimalDFTSize(image.shape[1])

    # Realiza o padding da imagem
    padded = cv2.copyMakeBorder(image, 0, dft_M - image.shape[0], 0, dft_N - image.shape[1], cv2.BORDER_CONSTANT, value=0)

    # Cria a matriz temporária para o filtro
    tmp = np.zeros((dft_M, dft_N), dtype=np.float32)

    gammaH = 15.0
    gammaL = 1.0
    c = 1.0
    D0 = 20

    cv2.namedWindow('Filtro Homomórfico')

    cv2.createTrackbar('gammaH', 'Filtro Homomórfico', int(gammaH * 10), 100, on_trackbar)
    cv2.createTrackbar('gammaL', 'Filtro Homomórfico', int(gammaL * 10), 100, on_trackbar)
    cv2.createTrackbar('c', 'Filtro Homomórfico', int(c * 100), 100, on_trackbar)
    cv2.createTrackbar('D0', 'Filtro Homomórfico', D0, 100, on_trackbar)

    while True:
        # Atualiza os parâmetros do filtro homomórfico
        gammaH = cv2.getTrackbarPos('gammaH', 'Filtro Homomórfico') / 10.0
        gammaL = cv2.getTrackbarPos('gammaL', 'Filtro Homomórfico') / 10.0
        c = cv2.getTrackbarPos('c', 'Filtro Homomórfico') / 100.0
        D0 = cv2.getTrackbarPos('D0', 'Filtro Homomórfico')

        # Preenche a matriz temporária
        for i in range(dft_M):
            for j in range(dft_N):
                tmp[i, j] = (gammaH - gammaL) * (1.0 - np.exp(-1.0 * c * ((((i - dft_M // 2) ** 2) + ((j - dft_N // 2) ** 2)) / (D0 ** 2)))) + gammaL

        # Calcula a DFT
        complexImage = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)

        # Realiza a troca de quadrantes
        deslocaDFT(complexImage)

        # Aplica o filtro frequencial
        filtered = cv2.mulSpectrums(complexImage, cv2.merge([tmp, tmp]), 0)

        # Troca novamente os quadrantes
        deslocaDFT(filtered)

        # Calcula a DFT inversa
        idft = cv2.idft(filtered, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

        filtered_image = cv2.normalize(idft, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        cv2.imshow('Imagem original', image)
        cv2.imshow('Imagem filtrada', filtered_image)
        cv2.imshow("Filtro", tmp)

        # Verifica se a tecla 'Esc' foi pressionada
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()