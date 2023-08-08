import cv2
import numpy as np
import torch

class BodyDetectionModel:
    def __init__(self, weights_path, confidence_threshold=0.5):
        """
        Initialize the BodyDetectionModel.

        Parameters:
            weights_path (str): Path to YOLOv5 model weights file.
            confidence_threshold (float, optional): Confidence threshold for detection. Default is 0.5.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("ultralytics/yolov5:v5.0", "yolov5s").autoshape()  # YOLOv5s model

        # Load weights
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold

    def detect_bodies(self):
        """
        Perform continuous body detection using webcam feed.
        """
        cap = cv2.VideoCapture(0)  # Use the default webcam (change the index if using an external camera)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            pred = results.pred[0]
            
            for det in pred:
                if det[4] >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, det[:4])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Body Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    model = BodyDetectionModel("yolov5s-seg.pt")
    model.detect_bodies()
