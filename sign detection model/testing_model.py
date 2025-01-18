"""import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_skills
hands = mp.hands.Hands(static_image_mode = False,min_detection_confidence = 0.3)

labels_dict = {i:chr(65+i) for i in range(26)}

while True:
    data_aux = []
    x_ =[]
    y_ = []
    
    ret,frame = cap.read()
    if not ret:
        break
    
    H,W = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

        for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
               
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_drawing.DrawingSpec(color = (0,255,0),circle_radius = 4),
            mp_drawing.DrawingSpec(color = (0,255,0),circle_radius = 2),
            
        )        
    x1 = int(min(x_)*W)- 10
    y1 = int(min(x_)*H)- 10
    x2 = int(min(x_)*W)+ 10
    y2 = int(min(x_)*H)+ 10
        
    if len(data_aux) == 42:
       prediction = model.predict([np.asanyarray(data_aux)]) 
       predicted_char = labels_dict[int(prediction[0])]
       
       cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
       cv2.putText(frame,predicted_char,(x1,y1-10),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(255,255,255))
       
cv2.imshow("sign Detection",frame)   

cap.release()
cv2.destroyAllWindows() """
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import os
import cv2
import mediapipe as mp

def load_test_data():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DATA_DIR = './data'  # Directory containing test images
    data = []
    labels = []

    # Iterate through each class directory
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue  # Skip if not a directory

        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)

            if img is None:
                print(f"Warning: Unable to read image {img_full_path}. Skipping.")
                continue  # Skip if the image cannot be read

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y
                        x_.append(x)
                        y_.append(y)

                    # Normalize the coordinates
                    if x_ and y_:  # Ensure there are landmarks
                        min_x = min(x_)
                        min_y = min(y_)
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(x_[i] - min_x)
                            data_aux.append(y_[i] - min_y)

                    data.append(data_aux)
                    labels.append(dir_)  # Ensure this corresponds to the correct class

    return np.array(data), np.array(labels)

def main():
    # Load the trained models
    with open('model.p', 'rb') as f:
        models = pickle.load(f)

    # Load new data for testing
    test_data, test_labels = load_test_data()  # Load your actual test data

    # Make predictions and evaluate each model
    for model_name, model in models.items():
        y_predict = model.predict(test_data)
        score = accuracy_score(test_labels, y_predict)
        print(f'{model_name} model accuracy: {score * 100:.2f}%')

if __name__ == "__main__":
    main()