import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

canvas = None
prev_x, prev_y = None, None

while True:
    success, frame = cap.read()
    if not success:
        break

   

    if canvas is None:
        canvas = np.zeros_like(frame)

    hands, frame = detector.findHands(frame, draw=True)

    if hands:
        lmList = hands[0]["lmList"]
        x, y = lmList[8][0], lmList[8][1]  # index finger tip

        if prev_x is None:
            prev_x, prev_y = x, y

        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    output = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    cv2.imshow("Gesture Whiteboard", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):      # quit
        break
    elif key == ord('c'):    # clear
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()

