import cv2

# Load the pre-trained Haar Cascades classifier for detecting human faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the video capture device
cap = cv2.VideoCapture(0)

# Define the line that the human heads need to cross
line_y = 300

# Initialize the head count
head_count = 0

# Loop over frames from the video capture device
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()

    # If the frame was not read successfully, break out of the loop
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the human faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw a line on the frame to indicate the line that the human heads need to cross
    cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 0), 2)

    # Loop over the detected faces and count the ones that cross the line
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # If the top of the detected face is below the line, increment the head count
        if y + h > line_y:
            head_count += 1

    # Display the frame and the head count
    cv2.imshow('Frame', frame)
    print(f'Head Count: {head_count}')

    # If the 'q' key is pressed, break out of the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
