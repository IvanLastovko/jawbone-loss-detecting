import re
import os
import argparse

import pandas as pd
from PIL import Image
from tqdm import tqdm


def to_json(document_path):
    with open(document_path, "r") as f:
        text = f.read()
    
    annotations = re.findall(
        rf"Enter Image Path:.*\n.*\n.*\n(.*):\s*Predicted in.*\n((?:.*%.*\n?)*)", 
        text
    )
    result = {"classes": [], "images": []}
    classes = []
    for path, labels in annotations:
        meta = {}
        path_meta = re.search(r"(.+)_(\d)+_(\d)+(\..+)", path)
        meta["origin_path"] = path_meta.group(1) + path_meta.group(4)
        meta["col_id"] = int(path_meta.group(2))
        meta["row_id"] = int(path_meta.group(3))

        image_path = os.path.join(
            os.path.dirname(document_path),
            path
        )
        image = Image.open(image_path)
        meta["width"], meta["height"] = image.size

        label_coords_list = re.findall(
            r"(.*):\s*\d+%\s*\(left_x:\s*(-?\d+)\s*top_y:\s*(-?\d+)\s*width:\s*(-?\d+)\s*height:\s*(-?\d+)\)", 
            labels
        )
        
        meta["annotations"] = []
        for label_coords in label_coords_list:
            c = label_coords[0]
            x, y, w, h = tuple(map(float, label_coords[1:]))
            try:
                class_id = result["classes"].index(c)
            except ValueError:
                class_id = len(result["classes"])
                result["classes"].append(c)
            x += w/2
            y += h/2
            meta["annotations"].append([class_id, x,y,w,h])

        result["images"].append(meta)
    
    return result


def save_as_yolo(json_data, save_dir, has_pieces):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    classes_path = os.path.join(save_dir, "classes.txt")
    with open(classes_path, "w") as dst:
        for c in json_data["classes"]:
            dst.write(c + "\n")
    
    for image in json_data["images"]:
        if has_pieces:
            path = os.path.basename(
                image["origin_path"].replace(
                    ".jpeg", f"_{image['col_id']}_{image['row_id']}.txt"
                )
            )
        else:
            path = os.path.basename(
                image["origin_path"].replace(".jpeg", ".txt")
            )
        annotation_path = os.path.join(save_dir, path)
        with open(annotation_path, "w") as dst:
            for label in image["annotations"]:
                label[1] /= image["width"]
                label[3] /= image["width"]
                label[2] /= image["height"]
                label[4] /= image["height"]
                label = list(map(strformat, label))
                label[0] = str(int(label[0]))
                dst.write(" ".join(label) + "\n")


def strformat(num):
    return "{:<08}".format(round(num, 6))


def join_data(json_data, original_width, original_height):
    result = {
        "classes": json_data["classes"],
        "images": []
    }
    df = pd.DataFrame(json_data["images"])
    groupped_images = df.groupby("origin_path")
    image_names = groupped_images.groups.keys()
    for path in tqdm(image_names, total=len(image_names)):
        pieces = groupped_images.get_group(path).sort_values(by=["col_id", "row_id"])
        image_meta = {
            "origin_path": path,
            "width": original_width,
            "height": original_height,
            "annotations": []
        }
        if pieces.shape[0] > 0:
            first_piece = pieces.iloc[0]
            for _, p in pieces.iterrows():
                for ann in p.annotations:
                    ann[1] += p.col_id * first_piece.width
                    ann[2] += p.row_id * first_piece.height
                    image_meta["annotations"].append(ann)

        result["images"].append(image_meta)
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for converting model's output to YOLO format."
    )
    parser.add_argument(
        "--document_path", "-d", dest="document_path",
        required=True, help="Path to document with predictions."
    )
    parser.add_argument(
        "--save_path", "-s", dest="save_path",
        required=True, help="Path to the directory where results will be stored."
    )
    parser.add_argument(
        "--join_pieces", "-j", dest="join_pieces",
        action="store_true", help="Join predictions from each image."
    )
    parser.add_argument(
        "--original_width", "-ow", dest="original_width",
        default=2560, type=int, help="Width of an original image."
    )
    parser.add_argument(
        "--original_height", "-oh", dest="original_height",
        default=1920, type=int, help="Height of an original image."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    json_data = to_json(args.document_path)
    has_pieces = True
    if args.join_pieces:
        json_data = join_data(json_data, args.original_width, args.original_height)
        has_pieces = False
    save_as_yolo(json_data, args.save_path, has_pieces)
