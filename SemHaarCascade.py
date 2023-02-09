import cv2
import numpy as np
from imutils import paths


def detectar():
    imagemPath = list(paths.list_images('imagens/'))
    numero = 0
    for z in imagemPath:
        img = cv2.imread(z)
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = cv2.medianBlur(img_cinza, 5)

        circles = cv2.HoughCircles(img2, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=20, minRadius=10, maxRadius=20)
        if isinstance(circles, np.ndarray):
            circles = np.uint16(np.around(circles))

            for i in circles[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 1)

        cv2.imshow("Sem Haar Cascade", img)
        cv2.waitKey(0)
        cv2.imwrite('IdentificadasSemHaar/' + str(numero) + '.png', img)
        numero += 1


detectar()
