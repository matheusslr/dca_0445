import cv2

def main():
    # Normal img
    img = cv2.imread(r'unidade 01\manipulando_pixels\imgs\chihiro.jpg')
    cv2.imshow('chihiro', img)

    rows, cols = img.shape[:2]

    start_point = (cols//4, rows//4)
    end_point = (cols - cols//4, rows - rows//4)

    # Invert img operation
    for i in range(start_point[0], end_point[0]):
        for j in range(start_point[1], end_point[1]):
            img[j][i] = 255 - img[j][i]

    cv2.imshow('inverted_chihiro', img)
    cv2.waitKey()


if __name__ == "__main__":
    main() 