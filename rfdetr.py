import cv2
import supervision as sv
from rfdetr import RFDETRBase,RFDETRNano
from rfdetr.util.coco_classes import COCO_CLASSES

model = RFDETRBase()
mask = cv2.imread('Mask1(Cashier).png', cv2.IMREAD_GRAYSCALE)
mask2 = cv2.imread('Mask2(Customer).png', cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture('Small.mp4')
while True:
    success, frame = cap.read()
    if not success:
        break
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask_resized2 = cv2.resize(mask2, (frame.shape[1], frame.shape[0]))
    masked_frame_cashier = cv2.bitwise_and(frame, frame, mask=mask_resized)
    masked_frame_customer = cv2.bitwise_and(frame, frame, mask=mask_resized2)

    detections = model.predict(masked_frame_cashier[:, :, ::-1].copy(), threshold=0.5)
    detections2 = model.predict(masked_frame_customer[:, :, ::-1].copy(), threshold=0.5)

    labels = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    labels2 = [
        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections2.class_id, detections2.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections)
    annotated_frame = sv.BoxAnnotator().annotate(annotated_frame, detections2)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections, labels)
    annotated_frame = sv.LabelAnnotator().annotate(annotated_frame, detections2,labels2)

    cv2.imshow("Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
