import base64
import json
import logging
import os
from pprint import pprint
from typing import Tuple

import cv2
import gdown
import numpy as np
import base64
import logging
import os
import urllib
import urllib.request

import calendar
import datetime


class Detector:

    def __init__(self) -> None:
        super(Detector
              ).__init__()

        # weights = "yolov4-tiny.weights"
        weights = "yolov4.weights"
        # cfg = "yolov4-tiny.cfg"
        cfg = "yolov4.cfg"
        self.labels = "coco.names"
        self.model = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.layers_names = self._get_layers_names()
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4

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

    def _bounding_box(self, points):
        x_coordinates, y_coordinates = zip(*points)

        return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]

    def _get_iou(self, ground_truth, pred):
        # coordinates of the area of intersection.
        ix1 = np.maximum(ground_truth[0], pred[0])
        iy1 = np.maximum(ground_truth[1], pred[1])
        ix2 = np.minimum(ground_truth[2], pred[2])
        iy2 = np.minimum(ground_truth[3], pred[3])

        # Intersection height and width.
        i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
        i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

        area_of_intersection = i_height * i_width

        # Ground Truth dimensions.
        gt_height = ground_truth[3] - ground_truth[1] + 1
        gt_width = ground_truth[2] - ground_truth[0] + 1

        # Prediction dimensions.
        pd_height = pred[3] - pred[1] + 1
        pd_width = pred[2] - pred[0] + 1

        area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

        iou = area_of_intersection / area_of_union

        return iou

    def _draw_bounding_box(self, img, x, y, x_plus_w, y_plus_h, num_max_cars=0, number_vehicles=0):

        point1, point2 = ((x + x_plus_w) // 2, ((y + y_plus_h) // 2))

        # cv2.polylines(img, [self.pts], True, (255, 0, 0), 2)

        cv2.circle(img, (int(point1), int(point2)), radius=0, color=(0, 255, 0), thickness=50)

        return img

    def inference(self, inputs: np.ndarray, pts: np.ndarray) -> np.array:
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
                pts = np.array(pts, np.int32)
                result = cv2.pointPolygonTest(pts, (point1, point2), measureDist=False)

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
            image_value = l["inputs"]
            if test is not None:
                for l in test:
                    # bbox = self._bounding_box(self.pts)
                    # print(bbox)
                    # pred = np.array([l["x"], l["y"], l["x_w"], l["y_h"]], dtype=np.float32)
                    # iou = self._get_iou(ground_truth=bbox, pred=pred)
                    # print(iou)
                    # if iou < 0.15:

                    image_value = self._draw_bounding_box(l["inputs"], l["x"], l["y"], l["x_w"], l["y_h"], total_carros,
                                                          3)

        return detections, total_carros, image_value

    def parking_cars(self, url: str, polygon: np.array) -> dict:
        date = datetime.datetime.utcnow()
        utc_time = calendar.timegm(date.utctimetuple())

        URL = f"{url}?{utc_time}"

        plate_url = urllib.request.urlopen(URL)
        plate_cloudnary = np.asarray(bytearray(plate_url.read()), dtype=np.uint8)
        plate_cloudnary = cv2.imdecode(plate_cloudnary, -1)

        tes, num_cars, image_r = self.inference(plate_cloudnary, polygon)

        cv2.imwrite("imagem_teste.jpg", image_r)

        image_base64 = base64.b64encode(cv2.imencode('.jpg', image_r)[1]).decode()

        dados = {
            'parked_cars': num_cars,
            'image': image_base64
        }

        return dados


if __name__ == '__main__':
    import base64
    import logging
    import os
    import urllib
    import urllib.request

    logger = logging.getLogger("Parking Detection")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    region = [
        [
            761,
            168
        ],
        [
            1050,
            110
        ],
        [
            1905,
            861
        ],
        [
            1554,
            1071
        ],
        [
            761,
            168
        ]
    ]

    detector = Detector()
    url = "http://187.37.89.204:88/jpg/image.jpg"
    result = detector.parking_cars(url=url, polygon=region)
    print(result)
