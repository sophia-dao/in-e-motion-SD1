import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained person detector (e.g., MobileNet SSD)
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

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
