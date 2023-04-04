import cv2

def main():
    # Normal img
    img = cv2.imread(r'exercicio_01\imgs\biel.png')
    cv2.imshow('biel', img)

    start_point = (50, 50)
    end_point = (150, 200)

    # Inverted img operation
    for i in range(start_point[0], end_point[0]):
        for j in range(start_point[1], end_point[1]):
            img[j][i] = 255 - img[j][i]

    cv2.imshow('inverted_biel', img)
    cv2.waitKey()


if __name__ == "__main__":
    main() 