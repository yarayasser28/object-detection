from ultralytics import YOLO
detection_model = YOLO("yolov8n.pt")
import cv2
import pyttsx3
engine = pyttsx3.init()
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf=conf)

    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            engine.say(f"{result.names[int(box.cls[0])]}")
            engine.runAndWait()
    return img, results
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:

    success, img = cap.read()

    if not success:
        break

    result_img, _ = predict_and_detect(detection_model, img, classes=[], conf=0.5)

    cv2.imshow("out", result_img)
    cv2.waitKey(1)

