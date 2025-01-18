import os
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up MediaPipe Hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define paths
input_folder = './data' 
output_folder = './annotated_data'  

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each subfolder in the input folder
for subdir, _, files in os.walk(input_folder):
    # Create a corresponding subdirectory in the output folder
    rel_subdir = os.path.relpath(subdir, input_folder)
    current_output_folder = os.path.join(output_folder, rel_subdir)
    os.makedirs(current_output_folder, exist_ok=True)

    # Process each image in the current subdirectory
    for image_name in files:
        image_path = os.path.join(subdir, image_name)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get hand landmarks
        results = hands.process(image_rgb)

        # Draw landmarks on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save the processed image with landmarks
        output_image_path = os.path.join(current_output_folder, f"landmark_{image_name}")
        cv2.imwrite(output_image_path, image)
        print(f"Saved image with landmarks: {output_image_path}")

# Release MediaPipe resources
hands.close()