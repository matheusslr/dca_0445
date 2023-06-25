# Processamento Digital de Imagens - DCA 0445
Olá, meu nome é Matheus Rodrigues! Este repositório é focado na resolução de exercícios da disciplina de Processamento Digital de Imagens da Universidade Federal do Rio Grande do Norte (UFRN), ministrato pelo professor 
[Agostinho Brito](https://agostinhobritojr.github.io/). Aqui você encontrará bastante OpenCV e referências do Studio Ghibli!

## Unidade 01: Processamento de Imagens no Domínio Espacial
### 1. Manipulando pixels em uma imagem
### 1.1. Negativo
O programa inicial solicita ao usuário as coordenadas e captura o negativo da área retangular. O algoritmo converte a imagem em tons de cinza e, para cada pixel na região, **calcula 255 menos o valor do pixel**.

<figure>
  <img
  src="unidade 01\manipulando_pixels\imgs\chihiro.jpg">
  <figcaption>Figura 1: Chihiro antes do negativo</figcaption>
</figure>

<figure>
  <img
  src="unidade 01\manipulando_pixels\imgs\inverted_chihiro.jpg">
  <figcaption>Figura 2: Chihiro depois do negativo</figcaption>
</figure>

A operação de inversão pode ser vista na parte do código abaixo: 
```python
# Invert img operation
for i in range(start_point[0], end_point[0]):
    for j in range(start_point[1], end_point[1]):
        img[j][i] = 255 - img[j][i]
```

### 1.2. Troca de regiões
Este código particiona a imagem em quatro quadrantes e troca suas respectivas regiões.

<figure>
  <img
  src="unidade 01\manipulando_pixels\imgs\totoro.jpg">
  <figcaption>Figura 3: Totoro antes da troca de regiões</figcaption>
</figure>

<figure>
  <img
  src="unidade 01\manipulando_pixels\imgs\swapped_totoro.jpg">
  <figcaption>Figura 4: Totoro depois da troca de regiões</figcaption>
</figure>

O segmento do código da troca de regiões pode ser visualizada abaixo:
```python
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
```

### 2. Decomposição de imagens em planos de bits
### 2.1. Esteganografia
De acordo com [(N. F. Johnson e S. Jajodia, 1998)](https://ieeexplore.ieee.org/document/4655281), a esteganografia é uma técnica que envolve ocultar um arquivo dentro de outro de forma criptografada. Ao contrário da criptografia, que busca tornar as mensagens incompreensíveis, o objetivo da esteganografia é esconder a existência de uma mensagem específica, camuflando-a dentro de arquivos, como imagens, músicas, vídeos ou textos. Com essa abordagem, é possível ocultar mensagens dentro de imagens, por exemplo, sem despertar suspeitas de que algo esteja escrito nelas.

No exemplo abaixo temos uma imagem contida em outra. Para descobrirmos a mensagem escondida dentro da imagem portadora usaremos operação bit a bit. Para isso, foi retirado os 5 bits mais significativos dos pixels da variável ```img_carrier``` e os 3 bits menos significativos serão alocados em uma nova variável ```img_encoded```.

<figure>
  <img
  src="unidade 01\decomposicao_img_bits\imgs\desafio-esteganografia.png">
  <figcaption>Figura 5: Imagem codificada</figcaption>
</figure>

<figure>
  <img
  src="unidade 01\decomposicao_img_bits\imgs\imagem_decodificada.png">
  <figcaption>Figura 6: Imagem escondida</figcaption>
</figure>

O codigo responsável pela decodificação pode ser visto abaixo:

```python
import cv2
import numpy as np

img = cv2.imread("unidade 01\decomposicao_img_bits\imgs\desafio-esteganografia.png")

if img is None:
    print("Erro ao abrir a imagem")
    exit()

img_carrier = np.copy(img)
img_encoded = np.copy(img)
nbits = 3

img_carrier = img >> nbits << nbits
img_encoded = img << (8 - nbits)

cv2.imshow("Imagem portadora", img_carrier)
cv2.imshow("Imagem codificada", img_encoded)
cv2.waitKey()
```

### 3. Preenchendo Regiões
### 3.1. Labeling
O objetivo deste programa é contar os objetos da imagem, distinguindo entre aqueles com e sem buracos.

Primeiro, temos a imagem:

<figure>
  <img
  src="unidade 01\preenchendo_regioes\imgs\bolhas.png">
  <figcaption>Figura 7: bolhas.png</figcaption>
</figure>

Após a leitura da Figura 7, é feito um pré-processamento, retirando as bolhas que tocam as bordas da imagem. O resultado é a Figura abaixo:

<figure>
  <img
  src="unidade 01\preenchendo_regioes\imgs\cropped_bolhas.png">
  <figcaption>Figura 8: cropped_bolhas.png</figcaption>
</figure>

O algoritmo responsável por essa parte pode ser visto abaixo.

```python
def is_object_on_edge(x, y, row, col):
    top_and_left = x == 0 or y == 0
    bottom_and_right = x == col - 1 or y == row - 1
    return top_and_left or bottom_and_right

  # Cropping objects on edges 
  for i in range(rows):
      for j in range(cols):
          if is_object_on_edge(j, i, rows, cols):
              cv2.floodFill(img, None, (j, i), 0)
```

Em seguida, é mudado o fundo da imagem para facilitar a contagem de buracos nas bolhas. A imagem resultante é a Figura 9.

<figure>
  <img
  src="unidade 01\preenchendo_regioes\imgs\new_background_bolhas.png">
  <figcaption>Figura 9: new_background_bolhas.png</figcaption>
</figure>

```python
# Chaging background to gray
cv2.floodFill(img, None, (0, 0), 133)
```

Logo depois, é feito a contagem de ocorrencias de bolhas e buracos contidos na imagem.

```python
# Looking for objects with and without holes
for i in range(rows):
    for j in range(cols):
        if img[i][j] == 255: 
            obj += 1
            cv2.floodFill(img, None, (j, i), obj)
        elif img[i][j] == 0:
            obj_holes += 1
            cv2.floodFill(img, None, (j, i), obj_holes)

print("Total of {} objects".format(obj))
print("Total of {} objects with hole".format(obj_holes))
print("Total of {} objects without hole".format(obj - obj_holes))
```
Saída:
```
Total of 21 objects
Total of 7 objects with hole
Total of 14 objects without hole
```