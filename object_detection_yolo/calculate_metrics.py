# https://github.com/yfpeng/object_detection_metrics/blob/master/podm/podm.py
import os
import sys
import argparse
from collections import Counter, OrderedDict
from enum import Enum
from typing import List, Dict

import pandas as pd
import numpy as np


class Box:
    def __init__(self, xtl: float, ytl: float, xbr: float, ybr: float):
        """
                    0,0 ------> x (width)
             |
             |  (Left,Top)
             |      *_________
             |      |         |
                    |         |
             y      |_________|
          (height)            *
                        (Right,Bottom)

        Args:
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
        """
        assert xtl <= xbr, f'xtl < xbr: xtl:{xtl}, xbr:{xbr}'
        assert ytl <= ybr, f'ytl < ybr: ytl:{ytl}, xbr:{ybr}'

        self.xtl = xtl
        self.ytl = ytl
        self.xbr = xbr
        self.ybr = ybr

    @property
    def width(self) -> float:
        return self.xbr - self.xtl

    @property
    def height(self) -> float:
        return self.ybr - self.ytl

    @property
    def area(self) -> float:
        return (self.xbr - self.xtl) * (self.ybr - self.ytl)

    def __str__(self):
        return 'Box[xtl={},ytl={},xbr={},ybr={}]'.format(self.xtl, self.ytl, self.xbr, self.ybr)

    @classmethod
    def intersection_over_union(cls, box1: 'Box', box2: 'Box') -> float:
        """
        Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between
        two bounding boxes.
        """
        # if boxes dont intersect
        if not Box.is_intersecting(box1, box2):
            return 0
        intersection = Box.intersection_area(box1, box2)
        union = Box.union_areas(box1, box2, intersection_area=intersection)
        # intersection over union
        iou = intersection / union
        assert iou >= 0, '{} = {} / {}, box1={}, box2={}'.format(iou, intersection, union, box1, box2)
        return iou

    @classmethod
    def is_intersecting(cls, box1: 'Box', box2: 'Box') -> bool:
        if box1.xtl > box2.xbr:
            return False  # boxA is right of boxB
        if box2.xtl > box1.xbr:
            return False  # boxA is left of boxB
        if box1.ybr < box2.ytl:
            return False  # boxA is above boxB
        if box1.ytl > box2.ybr:
            return False  # boxA is below boxB
        return True

    @classmethod
    def intersection_area(cls, box1: 'Box', box2: 'Box') -> float:
        xtl = max(box1.xtl, box2.xtl)
        ytl = max(box1.ytl, box2.ytl)
        xbr = min(box1.xbr, box2.xbr)
        ybr = min(box1.ybr, box2.ybr)
        # intersection area
        return (xbr - xtl) * (ybr - ytl)

    @staticmethod
    def union_areas(box1: 'Box', box2: 'Box', intersection_area: float = None) -> float:
        if intersection_area is None:
            intersection_area = Box.intersection_area(box1, box2)
        return box1.area + box2.area - intersection_area


class BoundingBox(Box):
    def __init__(self, image_name: str, label: str, xtl: float, ytl: float, xbr: float, ybr: float,
                 score=None):
        """Constructor.
        Args:
            image_name: the image name.
            label: class id.
            xtl: the X top-left coordinate of the bounding box.
            ytl: the Y top-left coordinate of the bounding box.
            xbr: the X bottom-right coordinate of the bounding box.
            ybr: the Y bottom-right coordinate of the bounding box.
            score: (optional) the confidence of the detected class.
        """
        super().__init__(xtl, ytl, xbr, ybr)
        self.image_name = image_name
        self.score = score
        self.label = label


class MetricPerClass:
    def __init__(self):
        self.label = None
        self.precision = None
        self.recall = None
        self.ap = None
        self.interpolated_precision = None
        self.interpolated_recall = None
        self.num_groundtruth = None
        self.num_detection = None
        self.tp = None
        self.fp = None

    @staticmethod
    def mAP(results: Dict[str, 'MetricPerClass']):
        return np.average([m.ap for m in results.values() if m.num_groundtruth > 0])


def calculate_metrics(
    ground_truth: List[BoundingBox],
    predictions: List[BoundingBox],
    iou_threshold: float,
) -> Dict[str, MetricPerClass]:
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Args:
        ground_truth: ground truth bounding boxes;
        predictions: detected bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5);
        method: It can be calculated as the implementation in the official PASCAL VOC toolkit (EveryPointInterpolation),
            or applying the 11-point interpolation as described in the paper "The PASCAL Visual Object Classes(VOC)
            Challenge" or AllPointsInterpolation" (ElevenPointInterpolation);
    Returns:
        A dictionary containing metrics of each class.
    """
    ret = {}  # list containing metrics (precision, recall, average precision) of each class

    # Get all classes
    classes = sorted(set(b.label for b in ground_truth + predictions))

    # Precision x Recall is obtained individually by each class
    # Loop through by classes
    for c in classes:
        preds = [b for b in predictions if b.label == c]  # type: List[BoundingBox]
        gt = [b for b in ground_truth if b.label == c]  # type: List[BoundingBox]
        npos = len(gt)

        tps = np.zeros(len(preds))
        fps = np.zeros(len(preds))

        # create dictionary with amount of gts for each image
        counter = Counter([os.path.basename(cc.image_name) for cc in gt])
        for key, val in counter.items():
            counter[key] = np.zeros(val)

        # Loop through detections
        for i in range(len(preds)):
            # Find ground truth image
            max_iou = sys.float_info.min
            mas_idx = -1
            for j in range(len(gt)):
                iou = Box.intersection_over_union(preds[i], gt[j])
                if iou > max_iou:
                    max_iou = iou
                    mas_idx = j
            # Assign detection as true positive/don't care/false positive
            if max_iou >= iou_threshold:
                if counter[os.path.basename(preds[i].image_name)][mas_idx] == 0:
                    tps[i] = 1  # count as true positive
                    counter[os.path.basename(preds[i].image_name)][mas_idx] = 1  # flag as already 'seen'
                else:
                    # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                    fps[i] = 1  # count as false positive
            else:
                fps[i] = 1  # count as false positive
        # compute precision, recall and average precision
        cumulative_fps = np.cumsum(fps)
        cumulative_tps = np.cumsum(tps)
        recalls = np.divide(cumulative_tps, npos, out=np.full_like(cumulative_tps, np.nan), where=npos != 0)
        precisions = np.divide(cumulative_tps, (cumulative_fps + cumulative_tps))
        ap, mpre, mrec, _ = calculate_all_points_average_precision(recalls, precisions)
        # add class result in the dictionary to be returned
        r = MetricPerClass()
        r.label = c
        r.precision = precisions
        r.recall = recalls
        r.ap = ap
        r.interpolated_recall = np.array(mrec)
        r.interpolated_precision = np.array(mpre)
        r.tp = np.sum(tps)
        r.fp = np.sum(fps)
        r.num_groundtruth = len(gt)
        r.num_detection = len(preds)
        ret[c] = r
    return ret


def calculate_all_points_average_precision(recall, precision):
    """
    All-point interpolated average precision

    Returns:
        average precision
        interpolated precision
        interpolated recall
        interpolated points
    """
    mrec = [0] + [e for e in recall] + [1]
    mpre = [0] + [e for e in precision] + [0]
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii


def convert_to_bbox(annotation_file):
    labels = pd.read_csv(annotation_file, sep=" ", header=None)
    labels.columns = ["cl", "x", "y", "w", "h"]
    bbox_list= []
    for _, label in labels.iterrows():
        xtl = label.x - label.w/2
        ytl = label.y - label.h/2
        xbr = label.x + label.w/2
        ybr = label.y + label.w/2

        bbox = BoundingBox(annotation_file, label.cl, xtl, ytl, xbr, ybr)
        bbox_list.append(bbox)
    
    return bbox_list


def find_bad_predictions(ground_truth_folder, predictions_folder, iou_th, bad_th):
    bad_predicitons = {}
    for f in os.listdir(predictions_folder):
        true_path = os.path.join(ground_truth_folder, f)
        pred_path = os.path.join(predictions_folder, f)

        try:
            true_bbox = convert_to_bbox(true_path)
            pred_bbox = convert_to_bbox(pred_path)
        except:
            continue
        
        cl_metrics = calculate_metrics(true_bbox, pred_bbox, iou_th)
        for metric in cl_metrics.values():
            if metric.ap < bad_th:
                bad_predicitons[f] = metric.ap
    
    return bad_predicitons


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for metric calculation for each image and printing all bad predictions."
    )
    parser.add_argument(
        "--ground_truth_folder", "-gt", dest="ground_truth_folder",
        required=True, help="Path to document with predictions."
    )
    parser.add_argument(
        "--predictions_folder", "-p", dest="predictions_folder",
        required=True, help="Path to the directory where results will be stored."
    )
    parser.add_argument(
        "--iou_th", "-iou", dest="iou_th",
        default=0.25, type=float, help="Intersection over union threshold."
    )
    parser.add_argument(
        "--bad_th", "-bad", dest="bad_th",
        default=0.25, type=float, help="Metric threshold for bad predictions."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    bad_predictions = find_bad_predictions(
        args.ground_truth_folder,
        args.predictions_folder,
        args.iou_th,
        args.bad_th
    )

    print(len(bad_predictions))
    print("Avg metric:", np.mean(list(bad_predictions.values())))
    for k, v in sorted(bad_predictions.items()):
        print(k, v)