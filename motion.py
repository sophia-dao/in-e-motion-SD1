import cv2
import mediapipe as mp
import numpy as np 

def list_available_cameras(max_cameras=10):
    available_cameras = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def select_capture_method():
    available_cameras = list_available_cameras()
    print("Select capture method:")
    for i, cam in enumerate(available_cameras):
        print(f"{i + 1}: Camera {cam}")
    print(f"{len(available_cameras) + 1}: Video file")
    choice = input("Enter the number of your choice: ")

    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(available_cameras):
            return cv2.VideoCapture(available_cameras[choice - 1])
        elif choice == len(available_cameras) + 1:
            video_file_path = input("Enter the path to the video file: ")
            return cv2.VideoCapture(video_file_path)
        else:
            print("Invalid choice. Defaulting to built-in camera.")
            return cv2.VideoCapture(0)
    else:
        print("Invalid input. Defaulting to built-in camera.")
        return cv2.VideoCapture(0)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Select the capture method
cap = select_capture_method()

# Check if the camera/video is opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Load pre-trained person detector (e.g., MobileNet SSD)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Prepare the frame for person detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    head_count = 0  # Initialize head count for this frame

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

                # Check if a head is detected (using nose landmark as proxy)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame[startY:endY, startX:endX], results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                    if nose_landmark.visibility > 0.5:  # Check visibility threshold
                        head_count += 1

    # Display the resulting frame
    cv2.imshow('Multi-Person Pose Estimation', frame)

    # Output to terminal
    print(f"Processed frame with {head_count} head(s) detected.")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
