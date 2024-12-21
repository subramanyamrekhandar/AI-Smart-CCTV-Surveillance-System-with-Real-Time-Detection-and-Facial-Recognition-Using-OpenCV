import cv2
import os
import numpy as np
import tkinter as tk
import tkinter.font as font

def collect_data():
    """Collects face data for training."""
    name = input("Enter name of person: ")
    ids = input("Enter ID: ")
    count = 1

    cap = cv2.VideoCapture(0)
    filename = "haarcascade_frontalface_default.xml"

    if not os.path.exists("persons"):
        os.makedirs("persons")  # Create the persons directory if it doesn't exist

    cascade = cv2.CascadeClassifier(filename)

    while True:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            # Save face image to persons folder
            cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
            count += 1

            cv2.putText(frm, f"Count: {count}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            cv2.imshow("New Face", roi)

        cv2.imshow("Collecting Data", frm)

        # Exit if 'Esc' key is pressed or 300 images are collected
        if cv2.waitKey(1) == 27 or count > 300:
            cv2.destroyAllWindows()
            cap.release()
            print("Data collection completed. Starting training...")
            train()
            break


def train():
    """Trains the face recognizer with collected data."""
    print("Training process initiated!")

    recog = cv2.face.LBPHFaceRecognizer_create()  # Requires opencv-contrib-python
    dataset = 'persons'

    paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]
    faces = []
    ids = []

    for path in paths:
        ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))
        faces.append(cv2.imread(path, 0))

    recog.train(faces, np.array(ids))
    recog.save('model.yml')
    print("Training completed successfully!")


def identify():
    """Identifies faces using the trained model."""
    cap = cv2.VideoCapture(0)
    filename = "haarcascade_frontalface_default.xml"

    if not os.path.exists("model.yml"):
        print("No trained model found! Please train the model first.")
        return

    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('model.yml')

    cascade = cv2.CascadeClassifier(filename)

    # Prepare labels for display
    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    labelslist = {}
    for path in paths:
        labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]

    print("Labels loaded:", labelslist)

    while True:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]

            label, confidence = recog.predict(roi)

            if confidence < 100:  # Confidence threshold
                cv2.putText(frm, f"{labelslist[str(label)]} ({int(confidence)})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frm, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frm)

        # Exit if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break


def maincall():
    """Creates the GUI for the application."""
    root = tk.Tk()
    root.geometry("480x150")
    root.title("Face Recognition")

    label = tk.Label(root, text="Select an Option Below")
    label.grid(row=0, columnspan=2)
    label_font = font.Font(size=20, weight='bold', family='Helvetica')
    label['font'] = label_font

    btn_font = font.Font(size=15)

    button1 = tk.Button(root, text="Add Member", command=collect_data, height=2, width=20)
    button1.grid(row=1, column=0, pady=(10, 10), padx=(5, 5))
    button1['font'] = btn_font

    button2 = tk.Button(root, text="Identify Member", command=identify, height=2, width=20)
    button2.grid(row=1, column=1, pady=(10, 10), padx=(5, 5))
    button2['font'] = btn_font

    root.mainloop()


if __name__ == "__main__":
    maincall()
