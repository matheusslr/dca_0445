import cv2

def main():
    img = cv2.imread(r'unidade 01\preenchendo_regioes\imgs\bolhas.png', cv2.COLOR_BGR2GRAY)

    rows, cols = img.shape[:2]
    obj = 0
    obj_holes = 0

    # Normal img
    cv2.imshow('bolhas', img)

    # Cropping objects on edges 
    for i in range(rows):
        for j in range(cols):
            if is_object_on_edge(j, i, rows, cols):
                cv2.floodFill(img, None, (j, i), 0)

    cv2.imshow('cropped_bolhas', img)

    # Chaging background to gray
    cv2.floodFill(img, None, (0, 0), 133)
    cv2.imshow('new_background_bolhas', img)

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

    cv2.waitKey()

def is_object_on_edge(x, y, row, col):
    top_and_left = x == 0 or y == 0
    bottom_and_right = x == col - 1 or y == row - 1
    return top_and_left or bottom_and_right

if __name__ == "__main__":
    main() 