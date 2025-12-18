import os
import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)


input_folder = './data' 
output_folder = './annotated_data'  


if not os.path.exists(output_folder):
    os.makedirs(output_folder)


for subdir, _, files in os.walk(input_folder):
    
    rel_subdir = os.path.relpath(subdir, input_folder)
    current_output_folder = os.path.join(output_folder, rel_subdir)
    os.makedirs(current_output_folder, exist_ok=True)

    
    for image_name in files:
        image_path = os.path.join(subdir, image_name)
        
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        results = hands.process(image_rgb)

        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        
        output_image_path = os.path.join(current_output_folder, f"landmark_{image_name}")
        cv2.imwrite(output_image_path, image)
        print(f"Saved image with landmarks: {output_image_path}")


hands.close()