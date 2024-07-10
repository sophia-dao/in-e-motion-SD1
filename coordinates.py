import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors

def list_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            available_cameras.append(index)
        cap.release()
    return available_cameras

def select_camera():
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras available.")
        exit()
    
    print("Available cameras:")
    for index in available_cameras:
        print(f"Camera index {index}")

    selected_index = int(input("Select camera index: "))
    if selected_index not in available_cameras:
        print("Invalid camera index.")
        exit()
    
    return selected_index

def print_detection_details(detections, frame_shape):
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame_shape[1], frame_shape[0], frame_shape[1], frame_shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            print(f"Detection {i}:")
            print(f" - Confidence: {confidence:.2f}")
            print(f" - Bounding Box: ({startX}, {startY}), ({endX}, {endY})")

def adjust_contrast(image, alpha=1.5, beta=0):
    """
    Adjust the contrast of the image.
    
    Parameters:
    - image: The input image.
    - alpha: Contrast control (1.0-3.0).
    - beta: Brightness control (0-100).
    
    Returns:
    - The image with adjusted contrast.
    """
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained person detector (e.g., MobileNet SSD)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Select the camera
camera_index = select_camera()
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Camera with index {camera_index} could not be opened.")
    exit()

# Get the screen resolution
screen = get_monitors()[0]
screen_width = screen.width
screen_height = screen.height

# Print the screen resolution
print(f"Screen resolution: {screen_width}x{screen_height}")

# Set the frame width and height to fit the screen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Print the set frame size
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Set frame size: {actual_width}x{actual_height}")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Adjust contrast of the frame
    frame = adjust_contrast(frame, alpha=1.5, beta=0)

    # Prepare the frame for person detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Print detection details
    print_detection_details(detections, frame.shape)

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            # Get the bounding box of the person
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box is within the frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(frame.shape[1], endX), min(frame.shape[0], endY)

            # Extract the person region from the frame
            person = frame[startY:endY, startX:endX]

            # Check if the person region is valid
            if person.size > 0:
                # Convert the person region to RGB
                rgb_person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)

                # Process the person region with MediaPipe Pose
                results = pose.process(rgb_person)

                # Draw pose landmarks on the original frame
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame[startY:endY, startX:endX], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Multi-Person Pose Estimation', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
