import os
import cv2
import mediapipe as mp
from collections import Counter

def count_valid_samples(data_dir="./annotated_data"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    class_counts = Counter()

    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Count only if landmarks are detected
            if results.multi_hand_landmarks:
                class_counts[dir_] += 1

    hands.close()
    return class_counts

if __name__ == "__main__":
    counts = count_valid_samples("./annotated_data")
    print("✅ Valid landmark samples per class:")
    for cls, cnt in counts.items():
        print(f"Class {cls}: {cnt} samples")
