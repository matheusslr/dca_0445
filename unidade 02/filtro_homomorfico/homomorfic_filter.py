import cv2
import numpy as np

def homomorphic_filter(image, cutoff_freq, c, gammaH, gammaL):
    # Convertendo a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Normalizando a imagem de entrada entre 0 e 1
    normalized = gray / 255.0
    
    # Aplicando a transformada de log para obter a imagem no domínio da frequência
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(normalized))))
    
    # Definindo os parâmetros do filtro homomórfico
    D0 = cutoff_freq  # Frequência de corte
    
    # Calculando a máscara do filtro homomórfico
    rows, cols = spectrum.shape
    center_row, center_col = int(rows / 2), int(cols / 2)
    y = np.arange(-center_row, rows - center_row)
    x = np.arange(-center_col, cols - center_col)
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    H = (gammaH - gammaL) * (1 - np.exp(-c * (D**2) / (D0**2))) + gammaL
    
    # Aplicando o filtro no domínio da frequência
    filtered_spectrum = np.exp(np.fft.ifft2(np.fft.ifftshift(spectrum * H)))
    
    # Normalizando a imagem filtrada para o intervalo [0, 255]
    filtered_image = np.uint8(filtered_spectrum.real * 255)
    
    return filtered_image, H

def on_cutoff_trackbar(value):
    global cutoff_freq
    cutoff_freq = value

def on_c_trackbar(value):
    global c
    c = value / 100.0

def on_gammah_trackbar(value):
    global gammaH
    gammaH = value / 100.0

def on_gammal_trackbar(value):
    global gammaL
    gammaL = value / 100.0

# Carregando a imagem
image = cv2.imread(r'unidade 02\filtro_homomorfico\img\ceu_escuro.jpg')

# Inicializando os valores dos parâmetros
cutoff_freq = 10
c = 1.0
gammaH = 1.5
gammaL = 0.5

# Criando a janela do OpenCV
cv2.namedWindow('Filtro Homomórfico')
cv2.namedWindow('Filtro')

# Criando os trackbars
cv2.createTrackbar('Cutoff', 'Filtro Homomórfico', cutoff_freq, 50, on_cutoff_trackbar)
cv2.createTrackbar('c', 'Filtro Homomórfico', int(c * 100), 200, on_c_trackbar)
cv2.createTrackbar('gammaH', 'Filtro Homomórfico', int(gammaH * 100), 200, on_gammah_trackbar)
cv2.createTrackbar('gammaL', 'Filtro Homomórfico', int(gammaL * 100), 200, on_gammal_trackbar)

while True:
    # Aplicando o filtro homomórfico
    filtered_image, H = homomorphic_filter(image, cutoff_freq, c, gammaH, gammaL)
    
    # Normalizando a imagem do filtro para o intervalo [0, 255]
    filtered_image_norm = (H - np.min(H)) / (np.max(H) - np.min(H))
    filtered_image_norm = np.uint8(filtered_image_norm * 255)
    
    # Exibindo a imagem original, a imagem filtrada e a imagem do filtro
    cv2.imshow('Imagem original', image)
    cv2.imshow('Imagem filtrada', filtered_image)
    cv2.imshow('Filtro', filtered_image_norm)
    
    # Verificando se a tecla 'Esc' foi pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()