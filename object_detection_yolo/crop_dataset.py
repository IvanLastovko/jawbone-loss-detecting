import os
import argparse

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def crop_data(dataset_path, save_dir, w_count, h_count):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    files = [f for f in os.listdir(dataset_path) if f.endswith(".png")]
    for filename in tqdm(files, total=len(files)):
        annotation_file = os.path.join(dataset_path, filename.replace(".png", ".txt"))
        
        if not os.path.exists(annotation_file):
            continue
        
        try:
            annotation = pd.read_csv(annotation_file, sep=" ", header=None)
        except pd.errors.EmptyDataError:
            annotation = pd.DataFrame([])

        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)
            continue
        height, width, _ = image.shape

        tile_w, tile_h = (1/w_count, 1/h_count)
        for i in range(w_count):
            border_xmin = i * tile_w
            border_xmax = border_xmin + tile_w
            for j in range(h_count):
                border_ymin = j * tile_h
                border_ymax = border_ymin + tile_h

                yolo_annotations = []
                for _, label in annotation.iterrows():
                    c, x, y, w, h = tuple(label.values)
                    c = int(c)
                    x_min = x - w/2
                    y_min = y - h/2
                    x_max = x + w/2
                    y_max = y + h/2

                    if (
                        border_xmin <= x < border_xmax and 
                        border_ymin <= y < border_ymax
                    ):
                        x_min = max(x_min, border_xmin)
                        y_min = max(y_min, border_ymin)
                        x_max = min(x_max, border_xmax)
                        y_max = min(y_max, border_ymax)

                        yolo_annotations.append(" ".join(map(str, [
                            c, 
                            strformat(min_max(x_min + (x_max - x_min)/2, border_xmin, border_xmax)), 
                            strformat(min_max(y_min + (y_max - y_min)/2, border_ymin, border_ymax)), 
                            strformat(min_max(x_max - x_min, 0, tile_w)), 
                            strformat(min_max(y_max - y_min, 0, tile_h))
                        ])))

                if len(yolo_annotations) == 0:
                    continue

                save_path = os.path.join(save_dir, filename.replace(".png", f"_{i}_{j}.txt"))
                with open(save_path, "w") as f:
                    f.write("\n".join(yolo_annotations))

                border_xmin_ = int(border_xmin * width)
                border_xmax_ = int(border_xmax * width)
                border_ymin_ = int(border_ymin * height)
                border_ymax_ = int(border_ymax * height)

                tile = image[border_ymin_:border_ymax_, border_xmin_:border_xmax_, :]
                cv2.imwrite(save_path.replace(".txt", ".png"), tile)

def strformat(num):
    return "{:<08}".format(round(num, 6))


def min_max(num, min_, max_):
    return (num - min_) / (max_ - min_)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for cropping images and annotations."
    )
    parser.add_argument(
        "--dataset_path", "-d", dest="dataset_path",
        required=True, help="Path to folder with images and annotations."
    )
    parser.add_argument(
        "--save_path", "-s", dest="save_path",
        required=True, help="Path to the directory where results will be stored."
    )
    parser.add_argument(
        "--x_divisions", "-x", dest="x_divisions",
        default=3, type=int, help="Number of divisons along the x-axis."
    )
    parser.add_argument(
        "--y_divisions", "-y", dest="y_divisions",
        default=2, type=int, help="Number of divisons along the y-axis."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    crop_data(
        args.dataset_path, 
        args.save_path, 
        args.x_divisions, 
        args.y_divisions
    )
