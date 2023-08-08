import cv2
import numpy as np

class BodyDetectionModel:
    def __init__(self, weights_path, config_path, class_names_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize the BodyDetectionModel.

        Parameters:
            weights_path (str): Path to YOLO pre-trained weights file.
            config_path (str): Path to YOLO model configuration file.
            class_names_path (str): Path to COCO class names file.
            confidence_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
            nms_threshold (float, optional): Non-maximum suppression threshold. Default is 0.4.
        """
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.classes = []
        with open(class_names_path, "r") as f:
            self.classes = f.read().splitlines()
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_bodies(self):
        """
        Perform continuous body detection using webcam feed.
        """
        cap = cv2.VideoCapture(0)  # Use the default webcam (change the index if using an external camera)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_names = self.net.getUnconnectedOutLayersNames()
            outputs = self.net.forward(layer_names)

            boxes = []
            confidences = []
            class_ids = []

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > self.confidence_threshold:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), font, 1, color, 2)

            cv2.imshow("Body Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    model = BodyDetectionModel("yolov3.weights", "yolov3.cfg", "coco.names")
    model.detect_bodies()
