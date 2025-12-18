import pickle
import numpy as np
import os
import cv2
import mediapipe as mp
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
    DATA_DIR = './annotated_data'
    data, labels = [], []

    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux, x_, y_ = [], [], []
                    for landmark in hand_landmarks.landmark:
                        x_.append(landmark.x)
                        y_.append(landmark.y)

                    if x_ and y_:
                        min_x, min_y = min(x_), min(y_)
                        for i in range(len(hand_landmarks.landmark)):
                            data_aux.append(x_[i] - min_x)
                            data_aux.append(y_[i] - min_y)

                    data.append(data_aux)
                    labels.append(dir_)

    return np.array(data), np.array(labels)

def main():
    with open('model.p', 'rb') as f:
        models = pickle.load(f)

    test_data, test_labels = load_test_data()

    for name, model in models.items():
        print(f"\n🔎 Evaluating {name}...")
        y_pred = model.predict(test_data)
        acc = accuracy_score(test_labels, y_pred)
        print(f"{name} test accuracy: {acc*100:.2f}%")

        # Confusion Matrix
        cm = confusion_matrix(test_labels, y_pred, labels=np.unique(test_labels))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=np.unique(test_labels),
                    yticklabels=np.unique(test_labels))
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(test_labels, y_pred))

if __name__ == "__main__":
    main()
