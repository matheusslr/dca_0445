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

O código da troca de regiões pode ser visualizada abaixo:
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

