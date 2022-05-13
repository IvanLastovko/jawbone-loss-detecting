import os
import glob
import argparse

from PIL import Image


def make_predictions(darknet_path, test_doc_path, dataset_path, config_path, weights_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(test_doc_path, "r") as fobj:
        darknet_script = os.path.join(darknet_path, "darknet")
        commands = [darknet_script, "detector test", dataset_path, config_path, weights_path]
        commands = " ".join(commands)
        image_List = [[num for num in line.split()] for line in fobj]
        for images in image_List:
            cmd = f"{commands} {images[0]}"
            os.system(cmd)

            prediciton_path = os.path.join(darknet_path, "predictions.jpg")
            predicted_image = Image.open(prediciton_path)

            output = os.path.join(save_path, os.path.basename(images[0]))
            predicted_image.save(output)
            os.remove(prediciton_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to predict bboxes and draw them on images."
    )
    parser.add_argument(
        "--darknet_path", "-dn", dest="darknet_path",
        default=".", help="Path to darknet folder."
    )
    parser.add_argument(
        "--test_doc_path", "-td", dest="test_doc_path",
        default="data/valid.txt", help="Path to document with data for testing."
    )
    parser.add_argument(
        "--dataset_path", "-ds", dest="dataset_path",
        default="data/obj.data", help="Path to dataset meta."
    )
    parser.add_argument(
        "--config_path", "-cfg", dest="config_path",
        default="cfg/yolo-obj.cfg", help="Path to model's config file."
    )
    parser.add_argument(
        "--weights_path", "-w", dest="weights_path",
        required=True, help="Path to model's weights."
    )
    parser.add_argument(
        "--save_path", "-s", dest="save_path",
        required=True, help="Path to the directory where results will be stored."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_predictions(
        args.darknet_path, 
        args.test_doc_path, 
        args.dataset_path,
        args.config_path, 
        args.weights_path,
        args.save_path
    )
