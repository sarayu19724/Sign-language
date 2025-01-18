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

    DATA_DIR = './annotated_data'  # Directory containing test images
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