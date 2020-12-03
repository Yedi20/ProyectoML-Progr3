import cv2
import os
import numpy as np
import time

def obtenerModelo(method, facesData, labels):
	if method == 'EigenFaces': emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
	if method == 'FisherFaces': emotion_recognizer = cv2.face.FisherFaceRecognizer_create()
	if method == 'LBPH': emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()

	#entrenar el reconocedor
	print('Entrenando ('+method+')...')
	inicio = time.time()
	emotion_recognizer.train(facesData, np.array(labels))
	tiempoEntrenamiento = time.time()-inicio
	print('tiempoEntrenamiento ('+method+'):',tiempoEntrenamiento)

	#Almacenamiento de modelo nobtenido
	emotion_recognizer.write('modelo'+method+'.xml')
   
dataPath = 'C:/Users/SAE/Desktop/ReconocimientoEmociones/data'    
#Cambia a la ruta donde hayas almacenado Data
emotionList = os.listdir(dataPath)
print('Lista de personas: ', emotionList)

labels = []
facesData = []
label = 0

for nameDir in emotionList:
    emotionPath = dataPath + '/' + nameDir
    #print('Leyendo las imágenes')
    
    for fileName in os.listdir(emotionPath):
        #print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(emotionPath+'/'+fileName,0))
        image = cv2.imread(emotionPath+'/'+fileName,0)
#v2.imshow('image',image)
#cv2.waitKey(10)
    label = label + 1

obtenerModelo('EigenFaces',facesData, labels)
obtenerModelo('FisherFaces',facesData, labels)
obtenerModelo('LBPH',facesData, labels)
#print('labels= ',labels)
#print('Número de etiquetas 0: ',np.count_nonzero(np.array(labels)==0))
#print('Número de etiquetas 1: ',np.count_nonzero(np.array(labels)==1))
#cv2.destroyAllWindows()
#python entrenandoRF.py
#face_recognizer = cv2.face.EigenFaceRecognizer_create()

#face_recognizer = cv2.face.FisherFaceRecognizer_create()

#entrenando el reconocedor de rosotos
#print('Entrenando...')
#face_recognizer.train(facesData, np.array(labels))

#almacenado los modelos obtendios
#face_recognizer.write('modeloEingenFace.xml')
#face_recognizer.write('modeloFisherFace.xml')
#print("Modelo almacenado")