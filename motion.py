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

def select_capture_methods():
    available_cameras = list_available_cameras()
    print("Select capture method(s) by entering the numbers separated by commas (e.g., 1,2):")
    for i, cam in enumerate(available_cameras):
        print(f"{i + 1}: Camera {cam}")
    print(f"{len(available_cameras) + 1}: Video file")
    choices = input("Enter your choices: ").split(',')

    captures = []
    for choice in choices:
        choice = choice.strip()
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(available_cameras):
                cap = cv2.VideoCapture(available_cameras[choice - 1])
                                                        # Set capture resolution:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust width
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1280) # Adjust height
                captures.append(cap)
            elif choice == len(available_cameras) + 1:
                video_file_path = input("Enter the path to the video file: ")
                captures.append(cv2.VideoCapture(video_file_path))
            else:
                print(f"Invalid choice: {choice}")
        else:
            print(f"Invalid input: {choice}")
    if not captures:
        print("No valid capture methods selected. Defaulting to built-in camera.")
        captures.append(cv2.VideoCapture(0))
    return captures

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Select the capture methods
caps = select_capture_methods()

# Check if the cameras/videos are opened successfully
for cap in caps:
    if not cap.isOpened():
        print("Error: Could not open one of the video sources.")
        exit()

# Load pre-trained person detector (using MobileNet SSD)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

total_head_count = 0  # total head count captured by all cameras

while True:
    frames = []
    for cap in caps:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from one of the cameras")
            continue
        frames.append(frame)

    new_head_count = 0

    for idx, frame in enumerate(frames):
        # Prepare the frame for person detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (568, 320), 100)
        net.setInput(blob)
        detections = net.forward()

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
                            new_head_count += 1

        # Resize the frame for display
        resized_frame = cv2.resize(frame, (1920,1080))  # Adjust width and height as needed

        # Display the resulting frame in a unique window
        cv2.imshow(f'Multi-Person Pose Estimation - Camera {idx + 1}', resized_frame)

    # Check if the total head count has changed
    if new_head_count != total_head_count:
        total_head_count = new_head_count
        print(f"Total head count updated: {total_head_count}")

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the captures
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
