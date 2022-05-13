import os
import shutil
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script that splits dataset into train/test subsets and prepares it for YOLO training."
    )
    parser.add_argument(
        "--source_path", "-src", dest="source_path",
        required=True, help="Path to folder with images and annotations."
    )
    parser.add_argument(
        "--dst_path", "-dst", dest="dst_path",
        required=True, help="Path to the directory where results will be stored."
    )
    parser.add_argument(
        "--class_name", "-n", dest="class_name",
        default="target", help="Target class name."
    )
    parser.add_argument(
        "--random_seed", "-rs", dest="random_seed",
        default=42, type=int, help="Random seed is needed for reproducability."
    )
    parser.add_argument(
        "--train_size", "-ts", dest="train_size",
        default=0.7, type=float, help="Percent of dataset which is used for training. Validation size will be 1 - [train_size]"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random_seed = args.random_seed
    source_path = args.source_path
    dst_path = args.dst_path
    train_size = args.train_size
    class_name = args.class_name

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    data_path = os.path.join(dst_path, "obj")

    shutil.move(source_path, dst_path)
    os.rename(
        os.path.join(dst_path, os.path.basename(source_path)),
        data_path
    )

    images = [f for f in os.listdir(data_path) if f.endswith(".png")]
    random.seed(random_seed)
    random.shuffle(images)

    last_train_idx = int(len(images) * train_size)

    with open(os.path.join(dst_path, "train.txt"), "w") as out:
        for img in images[:last_train_idx]:
            filename = os.path.join(data_path, img)
            out.write(filename + "\n")

    with open(os.path.join(dst_path, "valid.txt"), "w") as out:
        for img in images[last_train_idx:]:
            filename = os.path.join(data_path, img)
            out.write(filename + "\n")

    with open(f"{data_path}.data", "w") as out:
        out.write("classes = 1\n")
        out.write(f"train = {dst_path}/train.txt\n")
        out.write(f"valid = {dst_path}/valid.txt\n")
        out.write(f"names = {dst_path}/obj.names\n")
        out.write("backup = backup/")
    
    with open(f"{data_path}.names", "w") as out:
        out.write(f"{class_name}")
