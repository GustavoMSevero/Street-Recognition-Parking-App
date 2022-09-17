import json
import logging
import os

import cv2
import gdown
import numpy as np


class Detector:

    def __init__(self, polygon) -> None:
        super().__init__()

        weights = "yolov4.weights"
        cfg = "yolov4.cfg"
        self.labels = "coco.names"
        self.model = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers_names = self._get_layers_names()
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.pts = np.array(polygon, np.int32)

    @staticmethod
    def _load_labels(labels_path: str):

        with open(labels_path, 'r') as labels_file:
            labels = labels_file.read()
        labels = labels.split("\n")
        labels = list(map(lambda x: x.strip(), labels))
        labels = dict({k: v for k, v in enumerate(labels)})
        labels.update({-1: 'Unknown'})
        return labels

    def _get_layers_names(self):
        layers_names = self.model.getLayerNames()
        layers_names = [layers_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        return layers_names

    def _draw_bounding_box(self, img, x, y, x_plus_w, y_plus_h):

        point1, point2 = ((x + x_plus_w) // 2, ((y + y_plus_h) // 2))

        result = cv2.pointPolygonTest(self.pts, (point1, point2), measureDist=False)
        cv2.polylines(img, [self.pts], True, (255, 0, 0), 2)
        if result == 1.0:
            cv2.circle(img, (int(point1), int(point2)), radius=0, color=(0, 255, 0), thickness=50)

    def inference(self, inputs: np.ndarray) -> np.array:
        if not isinstance(inputs, np.ndarray) and len(inputs) == 1:
            inputs = inputs.pop()

        blob = cv2.dnn.blobFromImage(inputs, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.model.setInput(blob)
        layerOutputs = self.model.forward(self.layers_names)
        Width = inputs.shape[1]
        Height = inputs.shape[0]

        class_ids = []
        confidences = []
        boxes = []
        detections = []
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold,
                                   self.nms_threshold)

        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                detections.append((box, confidences[i], self.labels[class_ids[i]]))

                self._draw_bounding_box(inputs, round(x), round(y), round(x + w), round(y + h))

        return detections


if __name__ == '__main__':
    logger = logging.getLogger("Parking Detection")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    if os.path.isfile('yolov4.weights') != True or os.path.isfile('yolov4.weights') != True:
        logger.info("yolov4.weights will be downloaded...")

        url = 'https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'
        output = 'yolov4.weights'
        gdown.download(url, output, quiet=False)

        url_cfg = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
        output_cfg = "yolov4.cfg"
        gdown.download(url_cfg, output_cfg, quiet=False)

    with open("geometry.json") as file:
        region = json.load(file)

    detector = Detector(polygon=region["region"])
    image = cv2.imread("vant-externo.jpeg")

    tes = detector.inference(image)
    cv2.namedWindow("Street Parking", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Street Parking", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
