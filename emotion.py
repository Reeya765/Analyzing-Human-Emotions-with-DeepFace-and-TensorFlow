import cv2
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop to continuously capture frames from the webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Analyze emotions on the frame using DeepFace
    result = DeepFace.analyze(frame, actions=['emotion'])

    # Get the dominant emotion and its confidence score
    dominant_emotion = result[0]['dominant_emotion']
    confidence = result[0]['emotion'][dominant_emotion]

    # Display the detected emotion on the frame
    cv2.putText(frame, f'{dominant_emotion}: {confidence:.2f}%', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with emotion text
    cv2.imshow("Emotion Analyzer", frame)

    # Exit the webcam feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
