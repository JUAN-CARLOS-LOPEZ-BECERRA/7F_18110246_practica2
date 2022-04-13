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
        
def histo2(img1,img2,imr1,imr2, img1e,img2e,img1re, img2re):

    plt.subplot(2, 4, 1)
    plt.title("imagen1")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([img1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 2)
    plt.title("imagen2")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 3)
    plt.title("imagen resp")
    for i, c in enumerate(color):
        hist = cv2.calcHist([imr1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 4)
    plt.title("imagen resp")
    for i, c in enumerate(color):
        hist = cv2.calcHist([imr2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 5)
    plt.title("imagen1 equa")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img1e], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 6)
    plt.title("imagen2 equa")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img2e], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 7)
    plt.title("imagen1 resp equa")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img1re], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.subplot(2, 4, 8)
    plt.title("imagen2 resp equa")
    for i, c in enumerate(color):
        hist = cv2.calcHist([img2re], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    plt.show()
    
        
def ecual2(img, img2, imr1, imr2):
    img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resul1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagen1',resul1)

    img_to_yuv = cv2.cvtColor(img2,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resul2 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagen2',resul2)

    img_to_yuv = cv2.cvtColor(imr1,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resulr1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagenr1',resulr1)
    
    img_to_yuv = cv2.cvtColor(imr2,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    resulr2 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Equalizada imagenr2',resulr2)
    histo2(img, img2, imr1,imr2, resul1, resul2, resulr1,resulr2)
    
#suma met1
suma =img1+img2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
cv2.imshow('suma',suma)
ecual(img1, img2, suma)
cv2.destroyAllWindows() 

#suma met2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
suma=cv2.add(img1,img2)
cv2.imshow('suma',suma)
ecual(img1, img2, suma)
cv2.destroyAllWindows() 
#suma met3
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
suma=cv2.bitwise_or(img1,img2)
cv2.imshow('suma',suma)
ecual(img1, img2, suma)
cv2.destroyAllWindows() 

#resta met1
resta =img2-img1
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
cv2.imshow('resta',resta)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#resta met2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
resta=cv2.subtract(img1,img2)
cv2.imshow('resta',resta)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#resta met3



#multiplicacion met1
multi =img1*img2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
cv2.imshow('multiplicacion',multi)
ecual(img1, img2, suma)
cv2.destroyAllWindows()


#multiplicacion met2
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
multi=cv2.multiply(img1,img2)
cv2.imshow('multiplicacion',multi)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#multiplicacion met3
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
multi=cv2.bitwise_and(img1,img2)
cv2.imshow('multiplicacion',multi)
ecual(img1, img2, suma)
cv2.destroyAllWindows()


#division met 1 
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
div=cv2.divide(img1,img2)
cv2.imshow('division',div)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#division met 2 
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
div=img2/img1
cv2.imshow('division',div)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#potencia
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
res=cv2.pow(img1,2)
cv2.imshow('potencia1',res)
res2=cv2.pow(img2,2)
cv2.imshow('potencia2',res2)
ecual2(img1, img2, res, res2)
cv2.destroyAllWindows()


#OR
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
suma=cv2.bitwise_or(img1,img2)
cv2.imshow('OR',suma)
ecual(img1, img2, suma)
cv2.destroyAllWindows()
#AND
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
res=cv2.bitwise_and(img1,img2)
cv2.imshow('AND',res)
ecual(img1, img2, suma)
cv2.destroyAllWindows()

#NOT
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
res=cv2.bitwise_not(img1)
cv2.imshow('NOT1',res)
res2=cv2.bitwise_not(img2)
cv2.imshow('NOT2',res2)
ecual2(img1, img2, res, res2)
cv2.destroyAllWindows()


#traslacion
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
col = img1.shape[1] #columnas
fil = img1.shape[0] # filas
Img1= cv2.getRotationMatrix2D((col//2,fil//2),156,1)
res=cv2.warpAffine(img1,Img1,(col,fil))
cv2.imshow('traslacion 1',res)

col2 = img2.shape[1] #columnas
fil2 = img2.shape[0] # filas
Img2= cv2.getRotationMatrix2D((col2//2,fil2//2),156,1)
res2=cv2.warpAffine(img2,Img2,(col,fil))
cv2.imshow('traslacion 2',res2)
ecual2(img1, img2, res, res2)
cv2.destroyAllWindows()

#interpolcion
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
res= cv2.resize(img1,(400,300), interpolation=cv2.INTER_CUBIC)
cv2.imshow('interpolacion1',res)

res2= cv2.resize(img2,(400,300), interpolation=cv2.INTER_CUBIC)
cv2.imshow('interpolacion2',res2)
ecual2(img1, img2, res, res2)
cv2.destroyAllWindows()
'''
#logaritmo natural
#cv2.imshow('imagen1',img1)
#cv2.imshow('imagen2',img2)
#log1=cv2.log(img1,)
#log2=cv2.log(img2,)
#cv2.imshow('logaritmo natural1',log1)
#cv2.imshow('logaritmo natural2',log2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#raiz
cv2.imshow('imagen1',img1)
cv2.imshow('imagen2',img2)
div=cv2.sqrt(img1)
cv2.imshow('division',div)
key()'''






