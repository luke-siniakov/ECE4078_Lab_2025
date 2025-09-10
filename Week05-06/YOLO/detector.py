import cv2
import os
import numpy as np
from copy import deepcopy
from ultralytics import YOLO
from ultralytics.utils import ops


class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.class_colour = {
            'orange': (0, 165, 255),
            'lemon': (0, 255, 255),
            'pear': (0, 128, 0),
            'tomato': (0, 0, 255),
            'capsicum': (255, 0, 0),
            'potato': (255, 255, 0),
            'pumpkin': (255, 165, 0),
            'garlic': (255, 0, 255)
        }
        
        # Define confidence thresholds for each class
        # Adjusted based on common detection issues:
        self.class_confidence_thresholds = {
            'orange': 0.5,
            'lemon': 0.75,      # Higher threshold - reduces false positives on random objects
            'pear': 0.4,
            'tomato': 0.65,     # Higher threshold - reduces confusion with capsicum
            'capsicum': 0.65,   # Higher threshold - reduces confusion with tomato
            'potato': 0.7,      # Higher threshold - reduces false positives on random objects
            'pumpkin': 0.5,
            'garlic': 0.25      # Much lower threshold - helps detect garlic that's being missed
        }
        
        # Default threshold for classes not in the dictionary
        self.default_confidence_threshold = 0.9

    def detect_single_image(self, img):
        """
        function:
            detect target(s) in an image
        input:
            img: image, e.g., image read by the cv2.imread() function
        output:
            bboxes: list of lists, box info [label,[x,y,width,height],confidence] for all detected targets in image
            img_out: image with bounding boxes and class labels drawn on
        """
        bboxes = self._get_bounding_boxes(img)

        img_out = deepcopy(img)

        # draw bounding boxes on the image
        for bbox in bboxes:
            #  translate bounding box info back to the format of [x1,y1,x2,y2]
            xyxy = ops.xywh2xyxy(bbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), self.class_colour[bbox[0]], thickness=2)

            # draw class label with confidence score
            confidence = bbox[2] if len(bbox) > 2 else 0.0
            label_text = f"{bbox[0]} ({confidence:.2f})"
            img_out = cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.class_colour[bbox[0]], 2)

        return bboxes, img_out

    def _get_bounding_boxes(self, cv_img):
        """
        function:
            get bounding box and class label of target(s) in an image as detected by YOLOv8
            with class-specific confidence filtering
        input:
            cv_img: image, e.g., image read by the cv2.imread() function
        output:
            bounding_boxes: list of lists, box info [label,[x,y,width,height],confidence] for all detected targets in image
        """

        # predict target type and bounding box with your trained YOLO
        # Use a lower general confidence threshold since we'll filter by class-specific thresholds
        predictions = self.model.predict(cv_img, imgsz=320, verbose=False, conf=0.1)

        # get bounding box and class label for target(s) detected
        bounding_boxes = []
        for prediction in predictions:
            boxes = prediction.boxes
            for box in boxes:
                # Get class name and confidence
                box_label = box.cls  # class label of the box
                class_name = prediction.names[int(box_label)]
                confidence = float(box.conf[0])  # confidence score
                
                # Get class-specific threshold
                threshold = self.class_confidence_thresholds.get(class_name, self.default_confidence_threshold)
                
                # Only include detection if confidence meets class-specific threshold
                if confidence >= threshold:
                    # bounding format in [x, y, width, height]
                    box_cord = box.xywh[0]
                    bounding_boxes.append([class_name, np.asarray(box_cord), confidence])

        return bounding_boxes
    
    def set_class_confidence_threshold(self, class_name, threshold):
        """
        Set confidence threshold for a specific class
        """
        self.class_confidence_thresholds[class_name] = threshold
    
    def get_class_confidence_threshold(self, class_name):
        """
        Get confidence threshold for a specific class
        """
        return self.class_confidence_thresholds.get(class_name, self.default_confidence_threshold)


# FOR TESTING ONLY
if __name__ == '__main__':
    # get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    yolo = Detector(f'{script_dir}/model/best (1).pt')
    
    # Example: Modify confidence thresholds for specific classes
    yolo.set_class_confidence_threshold('tomato', 0.8)  # Make tomato detection more strict
    yolo.set_class_confidence_threshold('pear', 0.3)    # Make pear detection more lenient

    img = cv2.imread(f'{script_dir}/test/test_image_2.png')

    bboxes, img_out = yolo.detect_single_image(img)

    print("Detected objects:")
    for bbox in bboxes:
        class_name = bbox[0]
        confidence = bbox[2] if len(bbox) > 2 else "N/A"
        threshold = yolo.get_class_confidence_threshold(class_name)
        print(f"  {class_name}: confidence={confidence:.3f}, threshold={threshold}")
    
    print(f"Total detections: {len(bboxes)}")

    cv2.imshow('yolo detect', img_out)
    cv2.waitKey(0)