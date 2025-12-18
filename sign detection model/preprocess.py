import os
import pickle
import mediapipe as mp 
import cv2   

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './annotated_data'

data = []
labels = []

# Iterate through each directory and image
for dir_ in os.listdir(DATA_DIR):
    print(f"Processing directory: {dir_}")
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img_full_path = os.path.join(DATA_DIR, dir_, img_path)
        print(f"Processing image: {img_full_path}")
        
        img = cv2.imread(img_full_path)
        
        if img is None:
            print(f"Warning: Unable to read image {img_path}. Skipping.")
            continue
        
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
                labels.append(dir_)
                print(f"Detected landmarks for image: {img_path}")

        else:
            print(f"No hands detected in image: {img_path}")

# Save the data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data collection completed and saved successfully.")