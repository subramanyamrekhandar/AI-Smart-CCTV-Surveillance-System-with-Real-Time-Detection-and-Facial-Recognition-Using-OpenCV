import cv2

# Global variables
donel = False
doner = False
x1, y1, x2, y2 = 0, 0, 0, 0


def select(event, x, y, flags, param):
    """Mouse callback function to select region of interest."""
    global x1, x2, y1, y2, donel, doner
    if event == cv2.EVENT_LBUTTONDOWN:  # Left button to set the start point
        x1, y1 = x, y
        donel = True
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right button to set the end point
        x2, y2 = x, y
        doner = True
        print(f"Region selected: Top-left({x1}, {y1}) to Bottom-right({x2}, {y2})")


def rect_noise():
    """Main function to detect motion in a selected region of interest."""
    global x1, x2, y1, y2, donel, doner

    cap = cv2.VideoCapture(0)  # Open webcam

    # Ensure the camera is opened
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", select)

    try:
        # Step 1: Select region of interest
        print("Press 'Esc' after selecting the region with the mouse (Left-click for start, Right-click for end).")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            if donel:
                cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)  # Mark start point
            if doner:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw selected region

            cv2.imshow("Select Region", frame)

            # Exit loop if region is selected and user presses 'Esc'
            if cv2.waitKey(1) == 27 and donel and doner:
                cv2.destroyWindow("Select Region")
                break

        # Sort coordinates to ensure valid region selection
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # Validate selected region
        if x1 == x2 or y1 == y2:
            print("Error: Invalid region selected. Please try again.")
            return

        print("Region selected successfully. Monitoring motion...")

        # Step 2: Monitor motion in the selected region
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from camera.")
                break

            # Extract the region of interest (ROI)
            roi = frame[y1:y2, x1:x2]

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if prev_frame is None:
                prev_frame = gray
                continue

            # Calculate frame difference
            diff = cv2.absdiff(prev_frame, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            # Dilate to fill gaps
            thresh = cv2.dilate(thresh, None, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            motion_detected = False
            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Ignore small movements
                    continue
                motion_detected = True
                (mx, my, mw, mh) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x1 + mx, y1 + my), (x1 + mx + mw, y1 + my + mh), (0, 255, 0), 2)

            # Display motion status
            if motion_detected:
                cv2.putText(frame, "MOTION DETECTED", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "NO MOTION", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw the selected region on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow("Motion Detection", frame)

            # Update previous frame
            prev_frame = gray

            # Exit loop if 'Esc' is pressed
            if cv2.waitKey(1) == 27:
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()



# Test the function
if __name__ == "__main__":
    rect_noise()
