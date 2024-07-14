import cv2
import mediapipe as mp
import numpy as np
import multiprocessing

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

    capture_methods = []
    for choice in choices:
        choice = choice.strip()
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(available_cameras):
                capture_methods.append(('camera', available_cameras[choice - 1]))
            elif choice == len(available_cameras) + 1:
                video_file_path = input("Enter the path to the video file: ")
                capture_methods.append(('video', video_file_path))
            else:
                print(f"Invalid choice: {choice}")
        else:
            print(f"Invalid input: {choice}")
    if not capture_methods:
        print("No valid capture methods selected. Defaulting to built-in camera.")
        capture_methods.append(('camera', 0))
    return capture_methods

def draw_major_points(image, landmarks, connections, major_points, drawing_spec):
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            if idx in major_points:
                landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, image.shape[1], image.shape[0]
                )
                if landmark_px:
                    cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)

    if connections:
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx in major_points and end_idx in major_points:
                start_landmark = landmarks.landmark[start_idx]
                end_landmark = landmarks.landmark[end_idx]
                start_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    start_landmark.x, start_landmark.y, image.shape[1], image.shape[0]
                )
                end_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
                    end_landmark.x, end_landmark.y, image.shape[1], image.shape[0]
                )
                if start_px and end_px:
                    cv2.line(image, start_px, end_px, drawing_spec.color, drawing_spec.thickness)

def process_camera(camera_index, capture_method, major_points):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
    mp_drawing = mp.solutions.drawing_utils

    # Open the camera or video file
    if capture_method[0] == 'camera':
        cap = cv2.VideoCapture(capture_method[1])
    else:
        cap = cv2.VideoCapture(capture_method[1])

    if not cap.isOpened():
        print(f"Error: Could not open {capture_method[0]} {capture_method[1]}")
        return

    # Set capture resolution if desired
    if capture_method[0] == 'camera':
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Adjust width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Adjust height

    # Load pre-trained person detector (e.g., MobileNet SSD)
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

    total_head_count = 0  # Initialize total head count

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture image from {capture_method[0]} {camera_index}")
            continue

        new_head_count = 0  # Initialize new head count for this loop

        # Prepare the frame for person detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
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
                        draw_major_points(frame[startY:endY, startX:endX], results.pose_landmarks, mp_pose.POSE_CONNECTIONS, major_points, drawing_spec)
                        nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                        if nose_landmark.visibility > 0.5:  # Check visibility threshold
                            new_head_count += 1

        # Resize the frame for display
        resized_frame = cv2.resize(frame, (800, 600))  # Adjust width and height as needed

        # Display the resulting frame in a unique window
        cv2.imshow(f'Multi-Person Pose Estimation - Camera {camera_index}', resized_frame)

        # Check if the total head count has changed
        if new_head_count != total_head_count:
            total_head_count = new_head_count
            print(f"Camera {camera_index}: Total head count updated: {total_head_count}")

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Select the capture methods
    capture_methods = select_capture_methods()

    # Define the major key points
    major_points = [
        mp.solutions.pose.PoseLandmark.NOSE,
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
    ]

    # Create and start processes for each camera
    processes = []
    for idx, capture_method in enumerate(capture_methods):
        p = multiprocessing.Process(target=process_camera, args=(idx + 1, capture_method, major_points))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()
