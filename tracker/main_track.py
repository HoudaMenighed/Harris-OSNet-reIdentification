import numpy as np
import torch
import sys
import cv2
import gdown
from os.path import exists as file_exists, join
import torchvision.transforms as transforms

from .utils.nn_matching import NearestNeighborDistanceMetric
from .utils.detection import Detection
from .utils.tracker import Tracker

from .reid_multibackend import ReIDDetectMultiBackend

__all__ = ['TrackerMain']


class TrackerMain(object):
    def __init__(self,
                 model_weights,
                 device,
                 fp16,
                 cosine_threshold=0.2,
                 iou_threshold=0.7,
                 max_age=500, frames=3,
                 nn_=100,
                 lambda_=0.995,
                 alpha_=0.9
                 ):

        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        self.max_dist = cosine_threshold
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_dist, nn_)
        self.tracker = Tracker(
            metric, max_iou_distance=iou_threshold, max_age=max_age, n_init=frames)

    def update(self, bbox_xywh, confidences, classes, ori_img):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)
        # print('features : ', features)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes, confidences)

        # output bbox identities
        outputs = []

        for track in self.tracker.tracks:
            # print('track : ', track)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id, conf]))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs



    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]

            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

    def _get_features_h(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

            im[corners > 0.01 * corners.max()] = [0, 0, 255]
            im_crops.append(im)

        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

    def _get_features_s(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            sift = cv2.xfeatures2d.SIFT_create()

            keypoints, descriptors = sift.detectAndCompute(gray, None)

            im_with_keypoints = cv2.drawKeypoints(im, keypoints, None)

            im_crops.append(im_with_keypoints)

        if im_crops:
            features = self.model(im_crops)

        else:
            features = np.array([])
        return features
