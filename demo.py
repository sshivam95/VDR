# Import necessary libraries:
# cv2: OpenCV library for computer vision tasks
# numpy: Library for numerical operations
import cv2
import numpy as np

class BodyDetectionModel:
    def __init__(self, weights_path, config_path, class_names_path, confidence_threshold=0.5, nms_threshold=0.4):
        """
        Initialize the BodyDetectionModel with given parameters.
        
        Args:
        - weights_path (str): Path to YOLO pre-trained weights file.
        - config_path (str): Path to YOLO model configuration file.
        - class_names_path (str): Path to COCO class names file (list of object names that YOLO can detect).
        - confidence_threshold (float): Threshold for object detection. Only detections with confidence above this value are considered.
        - nms_threshold (float): Threshold for Non-maximum suppression (helps in removing redundant overlapping bounding boxes).
        """
        
        # Load the neural network using OpenCV's dnn module
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        # Load class names (e.g., "person", "car", etc.) from the provided file.
        with open(class_names_path, "r") as f:
            self.classes = f.read().splitlines()
        
        # Set confidence and nms thresholds
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

    def detect_bodies(self):
        """
        Perform continuous body detection using webcam feed.
        """
        
        # Initialize video capture with default webcam (index 0)
        cap = cv2.VideoCapture(0)  

        # Loop to continuously get frames from the webcam
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            # If frame is not read successfully, exit the loop
            if not ret:
                break

            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Convert the frame to a blob for neural network processing
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            
            # Get names of output layers from the neural network
            layer_names = self.net.getUnconnectedOutLayersNames()
            
            # Get detections from the neural network
            outputs = self.net.forward(layer_names)

            # Lists to store bounding boxes, confidences, and class IDs
            boxes = []
            confidences = []
            class_ids = []

            # Loop over each detection to extract information
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Filter out weak detections by ensuring the confidence meets the threshold
                    if confidence > self.confidence_threshold:
                        # Extract bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        # Append to our lists
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maximum suppression to suppress weak and overlapping bounding boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)

            # Define font and randomly generate colors for each class
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

            # Loop over the filtered detections to draw bounding boxes on the frame
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 10), font, 1, color, 2)

            # Display the frame with detections
            cv2.imshow("Body Detection", frame)
            
            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release video capture and destroy all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()


# Example usage of the class
if __name__ == "__main__":
    # Instantiate the model with YOLO weights, configuration, and class names
    model = BodyDetectionModel("yolov3.weights", "yolov3.cfg", "coco.names")
    
    # Start detecting bodies using the webcam feed
    model.detect_bodies()
