import cv2
import numpy as np
import sys

def swap_quadrants(image):
    tmp = np.empty_like(image)
    rows, cols = image.shape[:2]
    cx = cols // 2
    cy = rows // 2

    # Swap quadrants (Top-Left with Bottom-Right)
    tmp[:cy, :cx] = image[cy:, cx:]
    tmp[cy:, cx:] = image[:cy, :cx]
    tmp[cy:, :cx] = image[:cy, cx:]
    tmp[:cy, cx:] = image[cy:, :cx]

    np.copyto(image, tmp)

def main():
    if len(sys.argv) < 2:
        print("Uso: python nome_do_arquivo.py nome_da_imagem.png")
        return

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Erro abrindo imagem", sys.argv[1])
        return

    # Expand the input image to the optimal size for DFT
    dft_M = cv2.getOptimalDFTSize(image.shape[0])
    dft_N = cv2.getOptimalDFTSize(image.shape[1])
    padded = cv2.copyMakeBorder(image, 0, dft_M - image.shape[0], 0, dft_N - image.shape[1], cv2.BORDER_CONSTANT, value=0)

    # Prepare the complex matrix
    planes = [np.float32(padded), np.zeros(padded.shape, dtype=np.float32)]
    complex_image = cv2.merge(planes)

    # Perform DFT
    complex_image = cv2.dft(complex_image)
    swap_quadrants(complex_image)

    # Split the complex image into magnitude and phase
    planos = cv2.split(complex_image)

    # Calculate magnitude and phase spectrum
    magn, fase = cv2.cartToPolar(planos[0], planos[1], angleInDegrees=False)
    fase = cv2.normalize(fase, None, 0, 1, cv2.NORM_MINMAX)

    # Calculate magnitude spectrum only
    magn = cv2.magnitude(planos[0], planos[1])

    # Add a constant to avoid log(0)
    magn += 1

    # Calculate the logarithm of magnitude with dynamic range compression
    magn = np.log(magn)
    magn = cv2.normalize(magn, None, 0, 1, cv2.NORM_MINMAX)

    # Display the processed images
    cv2.imshow("Imagem", image)
    cv2.imshow("Espectro de magnitude", magn)
    cv2.imshow("Espectro de fase", fase)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
