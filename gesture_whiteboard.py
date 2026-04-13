import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


canvas = None
prev_x, prev_y = 0, 0
color = (0, 0, 255)  # default red
mode = "NONE"
thickness = 5


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    hands, frame = detector.findHands(frame, draw=True)

   
    cv2.rectangle(frame, (50, 0), (100, 50), (255, 0, 0), -1)   # Blue
    cv2.rectangle(frame, (120, 0), (170, 50), (0, 255, 0), -1)  # Green
    cv2.rectangle(frame, (190, 0), (240, 50), (0, 0, 255), -1)  # Red

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)

        x, y = lmList[8][0], lmList[8][1]

      
        if y < 60:
            if fingers[1] == 1 and fingers[2] == 1:
                if 50 < x < 100:
                    color = (255, 0, 0)
                    print("Blue selected")

                elif 120 < x < 170:
                    color = (0, 255, 0)
                    print("Green selected")

                elif 190 < x < 240:
                    color = (0, 0, 255)
                    print("Red selected")

       
        if fingers[1] == 1 and fingers[2] == 0:
            mode = "DRAW"

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            # 🔥 BRUSH SIZE CONTROL
            x1, y1 = lmList[4][0], lmList[4][1]   # thumb
            x2, y2 = lmList[8][0], lmList[8][1]   # index
            dist = math.hypot(x2 - x1, y2 - y1)

            thickness = int(np.interp(dist, [20, 150], [2, 20]))

            cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)
            prev_x, prev_y = x, y

       
        elif fingers[0] == 1 and sum(fingers[1:]) == 0:
            mode = "ERASE"
            cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
            prev_x, prev_y = 0, 0

      
        elif fingers[1] == 1 and fingers[2] == 1:
            mode = "STOP"
            prev_x, prev_y = 0, 0

        else:
            mode = "NONE"
            prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

   
    output = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)

   
    cv2.putText(output, f"Mode: {mode}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.putText(output, f"Brush: {thickness}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Gesture Whiteboard", output)

   
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('s'):
        cv2.imwrite("drawing.png", canvas)

cap.release()
cv2.destroyAllWindows()
