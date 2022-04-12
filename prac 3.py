import numpy as np
from matplotlib import pyplot as plt
import cv2
img1 = cv2.imread('imagen1.jpg')
img1=cv2.resize(img1,(300,400))
img2 = cv2.imread('imagen2.jpg')
img2=cv2.resize(img2,(300,400))

def histo(img1,img2,imr,img1e,img2e,imgre):

    plt.subplot(2, 3, 1)
    plt.title("imagen1")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([img1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    
    plt.subplot(2, 3, 2)
    plt.title("imagen2")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
   
    
    plt.subplot(2, 3, 3)
    plt.title("imagen resp")
    for i, c in enumerate(color):
        hist = cv2.calcHist([imr], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
        
    plt.subplot(2, 3, 4)
    plt.title("imagen1 equalizada")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img1e], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    
    plt.subplot(2, 3, 5)
    plt.title("imagen2 ecualizada")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img2e], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
   
    
    plt.subplot(2, 3, 6)
    plt.title("imagen resp ecualizada")
    for i, c in enumerate(color):
        hist = cv2.calcHist([imgre], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
   
    plt.show()
        
def ecual(img, img2, imr):
    img_to_yuv = cv2.cvtColor(img1,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resul1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagen1',resul1)

    img_to_yuv = cv2.cvtColor(img2,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resul2 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagen2',resul2)

    img_to_yuv = cv2.cvtColor(imr,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resul3 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagenr',resul3)

    histo(img, img2, imr, resul1, resul2, resul1)
        

#suma met1
suma =img1+img2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
cv2.imshow('suma',suma)
ecual(img1, img2, suma)




cv2.waitKey(0)
cv2.destroyAllWindows()
