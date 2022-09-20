import json
import logging
import os
from pprint import pprint

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

    def _draw_bounding_box(self, img, x, y, x_plus_w, y_plus_h, number_vehicles):

        point1, point2 = ((x + x_plus_w) // 2, ((y + y_plus_h) // 2))

        cv2.polylines(img, [self.pts], True, (255, 0, 0), 2)

        cv2.putText(image, f"{str(number_vehicles)} veiculos estacionados ", (1287, 950), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(img, (int(point1), int(point2)), radius=0, color=(0, 255, 0), thickness=50)

    def inference(self, inputs: np.ndarray) -> np.array:
        if not isinstance(inputs, np.ndarray) and len(inputs) == 1:
            inputs = inputs.pop()

        labels = self._load_labels(self.labels)

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

        num_cars = 0

        list_detections = list()

        if len(indices) > 0:
            for i in indices:
                box = boxes[i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]
                detections.append((box, confidences[i], labels[class_ids[i]]))

                list_vehicle = ["car", "motorcycle", "bus", "truck"]

                if labels[class_ids[i]] in list_vehicle:
                    num_cars = 1

                list_detections.append(
                    dict(
                        inputs=inputs,
                        x=round(x),
                        y=round(y),
                        x_w=round(x + w),
                        y_h=round(y + h),
                        number_vehicles=num_cars
                    )
                )

            test = list()
            total_carros = 0
            for l in list_detections:

                point1, point2 = ((l["x"] + l["x_w"]) // 2, ((l["y"] + l["y_h"]) // 2))

                result = cv2.pointPolygonTest(self.pts, (point1, point2), measureDist=False)

                if result == 1.0:
                    total_carros = total_carros + 1
                    test.append(
                        dict(
                            inputs=l["inputs"],
                            x=l["x"],
                            y=l["y"],
                            x_w=l["x_w"],
                            y_h=l["y_h"],
                            result=result
                        )
                    )

            for l in test:
                self._draw_bounding_box(l["inputs"], l["x"], l["y"], l["x_w"], l["y_h"], total_carros)

        pprint(f"Há {total_carros} estacionados na área selecionada")
        return detections, total_carros


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

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (351, 761)
    fontScale = 1

    color = (255, 0, 0)
    thickness = 2

    detector = Detector(polygon=region["region"])
    image = cv2.imread("vant-externo.jpeg")

    tes, num_cars = detector.inference(image)

    cv2.namedWindow("Street Parking", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Street Parking", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
