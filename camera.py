import cv2
import numpy as np
from detection import AccidentDetectionModel
from notification import AccidentNotification

notification = AccidentNotification()
model = AccidentDetectionModel("model.json", 'model_weights.h5')

font = cv2.FONT_HERSHEY_SIMPLEX


def start_application():
    video = cv2.VideoCapture(r'input_data\crash\crash_16.mp4')
    frame_width = 640
    frame_height = 480
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video', frame_width, frame_height)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (224, 224))
        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        # Detect accident
        if prob > 0.86:  # Threshold adjusted to 0.7 (70%)
            notification.play_beep()
            if pred == "crash":
                # Get latitude and longitude (example values)
                accident_latitude = 28.6139
                accident_longitude = 77.2090
                notification.notify_accident(frame, accident_latitude, accident_longitude)
                return "Accident detected"

        result_text = f"{pred} {round(prob * 100, 2)}%"
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, result_text, (10, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    start_application()