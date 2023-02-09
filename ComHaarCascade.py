import cv2
import numpy as np
from imutils import paths


def detectar():
    imagemPath = list(paths.list_images('imagens/'))
    numero = 0
    detector_face = cv2.CascadeClassifier('cascade.xml')
    for z in imagemPath:
        img = cv2.imread(z)
        img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        deteccoes = detector_face.detectMultiScale(img_cinza)
        for (x, y, l, a) in deteccoes:
            imgbb = img_cinza[y:y + a, x:x + l]
            circles = cv2.HoughCircles(imgbb, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=20, minRadius=10, maxRadius=20)
            if isinstance(circles, np.ndarray):
                circles = np.uint16(np.around(circles))

                for i in circles[0, :]:
                    cv2.circle(img, (x + i[0], y + i[1]), i[2], (0, 255, 0), 2)
                    cv2.circle(img, (x + i[0], y + i[1]), 2, (0, 0, 255), 2)
        cv2.imshow("Com Haar Cascade", img)
        cv2.waitKey(800)
        cv2.imwrite('IdentificadasHaar/' + str(numero) + '.png', img)
        numero += 1


detectar()
