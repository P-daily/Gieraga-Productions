import cv2
import numpy as np


def classify_parking_spots_infinite_loop(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Kolory w HSV (podobno lepiej)
        blue_lower = np.array([100, 150, 50])
        blue_upper = np.array([140, 255, 255])

        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])

        # Maski dla specjalsów
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)


        # Niepełonosprawni
        contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        spot_number = 1  # Numerowanie miejsc

        for cnt in contours_blue:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 500:  # Pomijanie małych obiektów
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'[{spot_number}] Niepelnosprawni', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 2)
                spot_number += 1

        # Kierowniki
        contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_yellow:
            x, y, w, h = cv2.boundingRect(cnt)
            if cv2.contourArea(cnt) > 500:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f'[{spot_number}] Kierownictwo', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 2)
                spot_number += 1

        # Zwykłe
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        contours_other, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_other:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 3.0 and 500 < cv2.contourArea(cnt) < 3000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'[{spot_number}] Zwykle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)
                spot_number += 1

        cv2.imshow('Parking Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

classify_parking_spots_infinite_loop('vid1.mp4')
