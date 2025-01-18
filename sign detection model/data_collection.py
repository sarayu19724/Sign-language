import os
import cv2

data_dir="./data"
if not os.path.exists(data_dir):
     os.makedirs(data_dir)
     
number_of_classes = 26
class_images = 500


cap = cv2.VideoCapture(0)
for j in range(number_of_classes):
    # Check if the directory exists, if not, create it
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))
      
    print('collecting data for class {}'.format(j))
    
    # Wait for user to press 'm' to start capturing images
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'press m to capture', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('m'):
            break
        
    counter = 0   
    # Capture and save images
    while counter < class_images:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1
        
# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
