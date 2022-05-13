import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


def strformat(num):
    return "{:<08}".format(round(num, 6))


def min_max(num, min_, max_):
    return (num - min_) / (max_ - min_)


def main(annotations_path, dataset_dir):
    annotations = pd.read_csv(annotations_path)
    # process every image
    files = annotations.filename.unique().tolist()
    for filename in tqdm(files, total=len(files)):
        # get image shape
        img_file = os.path.join(dataset_dir, filename)
        img_h, img_w = np.array(Image.open(img_file)).shape
        # extract all bboxes for the certain image
        image_annotations = annotations.loc[annotations.filename == filename]

        # convert all bboxes to yolo annotations
        yolo_annotations = []
        for _, bbox in image_annotations.iterrows():
            # check if empty
            if bbox.isna().any():
                break

            # map absolute values to [0...1]
            x_min = min_max(bbox.x, 0, img_w)
            y_min = min_max(bbox.y, 0, img_h)
            w = min_max(bbox.width, 0, img_w)
            h = min_max(bbox.height, 0, img_h)

            # save in yolo format
            yolo_annotations.append(" ".join(map(str, [
                0, # class lable
                strformat(x_min + w/2),
                strformat(y_min + h/2),
                strformat(w),
                strformat(h)
            ])))

        if len(yolo_annotations) == 0:
            continue
        
        # save yolo annotations in txt
        save_path = os.path.join(dataset_dir, filename.replace(".png", f".txt"))
        with open(save_path, "w") as f:
            f.write("\n".join(yolo_annotations))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for converting annotations to yolo format."
    )
    parser.add_argument(
        "--dataset_dir", "-d", dest="dataset_dir",
        required=True, help="Path to folder with images and annotations."
    )
    parser.add_argument(
        "--annotation_path", "-a", dest="annotation_path",
        required=True, help="Path to the annotations csv file."
    )
    args = parser.parse_args()
    main(args.annotation_path, args.dataset_dir)
