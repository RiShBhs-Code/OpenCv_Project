import cv2
from tensorflow.keras.models import load_model
import numpy as np

url = 'https://192.168.23.44:8080'

model = load_model('C:/Users/rishb/Python_3,12.x_Projects/Dry_Wet_classifier.h5')

class_names = ['Dry Waste' , 'Wet Waste']

img_size = 224

cap = cv2.VideoCapture(url)

if not cap.isOpened() :
    print('Unable to Open Webcam :')
    exit(0)

while True :
    ret , frame = cap.read()

    if not ret :
        print('Unable to Capture the Frame , Exiting......')
        break  

image = cv2.resize(frame , (img_size , img_size))

predictions = model.predict(image)
class_ids = np.argmax(predictions)
label = class_names(class_ids)

confidence = predictions[class_ids]

text = f"{label}({confidence*100:.2f}%)"

cv2.putText(frame , text , (10 , 30 ) , cv2.FONT_HERSHEY_COMPLEX , 0.6 , (0 , 255 ,0 ) , 2)

cv2.imshow('Garbage classifier ' , frame)

if cv2.waitKey(1) & 0xFF == ord('q'):
    exit(0)

cap.release
cv2.destroyAllWindows 
